---
name: ultralytics-troubleshoooting
description: A concise, action-oriented troubleshooting playbook for code agents working with Ultralytics YOLO (install, runtime, training, prediction, export, environment). Optimized for automated diagnosis and repair.
license: AGPL-3.0
metadata:
    author: Burhan-Q
    version: "1.0"
    ultralytics-version: ">=8.4.0"
---

# Ultralytics Troubleshooting – Agent Skill

Purpose: enable a code agent to quickly diagnose and resolve common Ultralytics YOLO issues across install, runtime, training, prediction, export, and environment setup.

See also: docs reference links at the end for deeper guidance.

## When To Use

- New or broken setup: imports fail, CLI not found, CUDA issues.
- Training problems: slow/NaN loss, CPU-only, bad metrics, dataset errors.
- Prediction problems: no/empty results, wrong classes, plotting errors.
- Export problems: ONNX/TensorRT/OpenVINO conversion failures.

## Preconditions

- Python 3.8+ available.
- Internet access for package install/update (unless using local editable install).
- Python virtual environment (highly preferred)
- If GPU expected: correct CUDA + PyTorch build alignment.

## Inputs (Agent)

- User intent: task (install, train, predict, export), target device (cpu/gpu), environment (pip/conda/docker/headless).
- Paths: dataset YAML or folder, model weights/config, source media.
- Constraints: offline mode, limited permissions, containerized, macOS/Linux/Windows.

## Outputs (Agent)

- A minimal, reproducible command that works (CLI or Python).
- Environment summary and resolution notes.
- Pinned versions if needed, and links to generated artifacts (runs/, exported models).

## Safety / Side Effects

- Installing/upgrading packages may change environment behavior.
- Prefer editable install only when developing inside this repo.
- Avoid deleting user data; create new files under runs/ or temp dirs.

---

## Quick Decision Tree

1. Is Python virtual environment in use?
    - No → create virtual environment and install package.
    - Yes → Ensure virtual environment is active and package is installed.
2. Can `yolo version` or `python -c "import ultralytics as u; print(u.__version__)"` run?
    - No → Fix installation (uv/pip/conda or `pip install -e .`).
3. Expect GPU? Run `yolo checks` or try `python -c import torch;torch.cuda.is_available()`.
    - If False but GPU expected → align PyTorch ↔ CUDA; consider conda bundle.
4. Sanity test on tiny dataset (COCO8) succeeds?
    - No → inspect dataset path/format, permissions, args.
5. Prediction works on a sample image?
    - No → validate model path, task mismatch, classes filter, conf/imgsz.
6. Export works to ONNX?
    - No → update packages, try smaller imgsz/batch, verify opset, test Python API.

---

## Core Workflow (Agent Steps)

### 1) Preflight Checks

Run quick environment probes and capture output for the user:

```bash
python -c "import sys, ultralytics, torch;\nprint('py', sys.version);\nprint('ultralytics', ultralytics.__version__);\nprint('torch', torch.__version__);\nprint('cuda_available', torch.cuda.is_available())"

yolo version || true
yolo checks || true
yolo settings || true
```

Interpretation:

- If import fails → go to Install/Repair.
- If `cuda_available` is False but GPU expected → go to GPU Fix.
- Keep outputs in the final report.

### 2) Install / Repair

- Pip (stable):
    ```bash
    pip install -U ultralytics
    ```
- Pip (latest-main):
    ```bash
    pip install -U "git+https://github.com/ultralytics/ultralytics.git@main"
    ```
- Local editable (inside this repo):
    ```bash
    pip install -e .
    ```
- Headless servers (avoid libGL issues):
    ```bash
    pip install ultralytics-opencv-headless
    ```
- Conda (GPU bundle example):
    ```bash
    conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
    ```

Re-run Preflight after changes.

### 3) GPU Fix (if needed)

- Verify compatible PyTorch/CUDA pair per platform guidance (install PyTorch first for your CUDA).
- On older GPUs (SM < 7.5) newer cuDNN may be incompatible; use a PyTorch build matching older CUDA/cuDNN.
- Minimal probe the agent can run (optional):
    ```bash
    python - << 'PY'
    import torch
    print('cuda_available', torch.cuda.is_available())
    if torch.cuda.is_available():
    		print('device_cap', torch.cuda.get_device_capability(0))
    		print('cudnn_version', torch.backends.cudnn.version())
    PY
    ```

### 4) Sanity Test (Tiny Train/Val)

Run a known-good command to validate end-to-end:

```bash
yolo train model=yolo26n.pt data=coco8.yaml epochs=1 imgsz=320 batch=4 device=auto
```

Validate outputs under `runs/train/*` and include summary metrics in the report.

### 5) Dataset & Config Checks

- Verify dataset YAML path is correct and readable.
- Confirm labels/structure match task (detect/segment/pose/obb/classify).
- If unsure, attempt a validation-only pass to surface format errors:
    ```bash
    yolo val model=yolo26n.pt data=path/to/data.yaml imgsz=640 batch=1
    ```
- For custom training, try pretrained weights first to isolate data vs. optimization issues.

### 6) Training Issues

- Slow or CPU-only: set `device=0` explicitly or fix GPU (Step 3).
- Divergence/NaN: reduce `lr0`, lower `batch`, check labels, try fewer augmentations.
- Poor metrics: inspect class balance, increase epochs, verify annotations, adjust imgsz.
- Track metrics via TensorBoard/Comet if configured.

Example tweak:

```bash
yolo train model=yolo26n.pt data=... epochs=50 imgsz=640 batch=-1 lr0=0.005 device=0
```

- NOTE: `batch=-1` will auto-calculate batch size

### 7) Prediction Issues

- Ensure task ↔ weights match (e.g., `yolo26n-seg.pt` for segmentation).
- Use `classes`, `conf`, `imgsz` for filtering/quality:

```bash
yolo predict model=yolo26n.pt source=path/or/url imgsz=640 conf=0.25 classes=0,2 show=False save=True
```

- If blank outputs: lower `conf`, verify source path and image channels, try a canonical test image.

### 8) Export Issues

Start with ONNX; reduce complexity if needed:

```bash
yolo export model=yolo26n.pt format=onnx imgsz=640 dynamic=False opset=12
```

If failure persists:

- Update ultralytics, onnx, onnxruntime.
- Try Python API export:

    ```python
    from ultralytics import YOLO

    YOLO("yolo26n.pt").export(format="onnx", imgsz=640)
    ```

### 9) Environment Modes

- macOS: CPU by default; use Metal/Accelerate or external GPU not supported by CUDA.
- Linux servers: prefer headless build and/or Docker for reproducibility.
- Docker quick start (GPU):
    ```bash
    t=ultralytics/ultralytics:latest
    sudo docker pull $t
    sudo docker run -it --ipc=host --runtime=nvidia --gpus all $t
    ```

---

## Minimal Python Sanity (Optional)

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=1, imgsz=320)
res = model("https://ultralytics.com/images/bus.jpg")
print(res[0].summary())
```

## What To Return To The User

- Environment summary (py/ultralytics/torch versions, cuda availability, `yolo checks`).
- The working command(s) used and where results are stored (runs/\*).
- Any constraints or remaining gaps (e.g., GPU not present, export format limitations).

## References (Docs)

- Install & CLI basics: docs/en/quickstart.md
- Common issues guide: docs/en/guides/yolo-common-issues.md
- CLI usage & args: docs/en/usage/cli.md
- Python usage: docs/en/usage/python.md
- Docker quickstart: docs/en/guides/docker-quickstart.md
- Conda quickstart: docs/en/guides/conda-quickstart.md
- Datasets overview: docs/en/datasets/index.md
- Training tips: docs/en/guides/model-training-tips.md
