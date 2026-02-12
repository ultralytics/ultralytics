---
name: ultralytics-troubleshooting
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
- Docker installed (optional but helpful for environment isolation).
- Internet access for package install/update (unless using local editable install).
- Python virtual environment (highly preferred)
- If GPU expected: correct CUDA + PyTorch build alignment.

## Inputs (Agent)

- User intent: task (install, train, predict, export), target device (cpu/gpu), environment (uv/pip/conda/docker/headless).
- Paths: dataset YAML or folder, model weights/config, source media.
- Constraints: offline mode, limited permissions, containerized, macOS/Linux/Windows.

## Outputs (Agent)

- A minimal, reproducible command that works (CLI or Python).
- Environment summary and resolution notes, or all attempted commands.
- Pinned versions if needed, and links to generated artifacts (runs/, exported models).

## Safety / Side Effects

- Check with user before taking action, never assume intent.
- Verify with user before modifying files or installing packages.
- Avoid unnecessary package updates or environment changes.
- Prefer non-destructive diagnostics (e.g., `yolo checks`) before reinstalling or changing configs.
- Installing/upgrading packages may change environment behavior.
- Prefer editable install only when developing inside this repo.
- Avoid deleting user data; create new files under runs/ or temp dirs.

---

## Quick Decision Tree

1. Using containerized environment (Docker)?
    - Yes → Use Docker image and commands; skip to Step 3 for GPU checks.
    - No → Proceed to Step 2.
2. Is Python virtual environment in use?
    - No → See [ENVIRONMENT_SETUP.md](references/ENVIRONMENT_SETUP.md) for environment setup.
    - Yes → Ensure virtual environment is active and package is installed.
3. Can `yolo version` or `python -c "import ultralytics as u; print(u.__version__)"` run?
    - No → See [ENVIRONMENT_SETUP.md](references/ENVIRONMENT_SETUP.md) for installation guidance.
4. Expect GPU? Run `yolo checks` or try `python -c import torch;torch.cuda.is_available()`.
    - If False but GPU expected → align PyTorch ↔ CUDA; consider conda bundle.
5. Sanity test on tiny dataset (COCO8) succeeds?
    - No → inspect dataset path/format, permissions, args.
6. Prediction works on a sample image?
    - No → validate model path, task mismatch, classes filter, conf/imgsz.
7. Export works to ONNX?
    - No → update packages, try smaller imgsz/batch, verify opset, test Python API.

---

## Core Workflow (Agent Steps)

### 1) Preflight Checks

**If ultralytics is already installed:**

Run built-in diagnostics:

```sh
yolo checks
```

If this succeeds, proceed to the environment diagnostic below.

**If `yolo checks` fails or ultralytics is not installed:**

The system environment may not be properly configured. See [ENVIRONMENT_SETUP.md](references/ENVIRONMENT_SETUP.md) for detailed guidance on:
- Checking Python installation and version
- Creating and activating virtual environments
- Installing ultralytics with various package managers
- Troubleshooting installation issues

After completing environment setup from the reference guide, return here to continue.

**Environment diagnostic:**

Once installation is verified, run the `ultralytics` utility checks command:

```sh
yolo checks
```

Interpretation:

- If command fails/not-found → see [ENVIRONMENT_SETUP.md](references/ENVIRONMENT_SETUP.md).
- If no GPU found, but GPU expected → go to GPU Fix (Step 3).
- If packages are outdated or show problem (denoted with ❌) → go to Install/Repair (Step 2).
- Keep outputs in the final report.

### 2) Install / Repair

For detailed installation guidance including environment setup, package manager selection, and troubleshooting, see [ENVIRONMENT_SETUP.md](references/ENVIRONMENT_SETUP.md).

**Quick reference for common installation methods:**

If `uv` is not installed, use `pip` instead by removing preceding `uv` from the commands below.

- Install latest stable release:
  ```sh
  uv pip install ultralytics --upgrade
  ```

- **Latest development version:**
  ```sh
  uv pip install "git+https://github.com/ultralytics/ultralytics.git@main"
  ```

- **Editable install** (for `ultralytics` development):
  ```sh
  uv pip install -e ".[dev]"
  ```

- **Headless** (servers without display, uncommon):
  ```sh
  uv pip install ultralytics-opencv-headless
  ```

- **Conda** (GPU bundle, ONLY when `conda` explicitly preferred by user):
    **IMPORTANT**: PyTorch>2.5.1 is NOT available using `conda`
    ```sh
    conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
    ```
    **NOTE**: Adjust `pytorch-cuda` version to match the CUDA installation. If unclear, try running `nvidia-smi` and check the output for the CUDA version.

**After installation:** Re-run Preflight checks (Step 1) to verify proper installation.

### 3) GPU Fix (if needed)

- Verify compatible PyTorch/CUDA pair per platform guidance (install PyTorch first for your CUDA).
- On older GPUs (SM < 7.5) newer cuDNN may be incompatible; use a PyTorch build matching older CUDA/cuDNN.
- Minimal probe the agent can run (optional):
    ```sh
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

```sh
yolo train model=yolo26n.pt data=coco8.yaml epochs=1 imgsz=320 batch=1
```

Validate outputs under `runs/train/*` and include summary metrics in the report.

### 5) Dataset & Config Checks

- Verify dataset YAML path is correct and readable.
- Confirm labels/structure match task (detect/segment/pose/obb/classify).
- If unsure, attempt a validation-only pass to surface format errors:
    ```sh
    yolo val model=yolo26n.pt data=path/to/data.yaml imgsz=640 batch=1
    ```
- For custom training, try pretrained weights first to isolate data vs. optimization issues.

### 6) Training Issues

- Slow or CPU-only: set `device=0` explicitly or fix GPU (Step 3).
- Divergence/NaN: reduce `lr0`, lower `batch`, disable automatic-mixed precision `amp=False`, check labels, try fewer augmentations.
- Poor metrics: inspect dataset class balance, increase epochs, verify annotations, adjust `imgsz`.
- Track metrics via Ultralytics Platform, TensorBoard, W&B, Comet, etc. if configured.

Example tweak:

```sh
yolo train model=yolo26n.pt data=... epochs=50 imgsz=640 batch=-1 lr0=0.005 device=0
```

- NOTE: `batch=-1` will auto-calculate batch size

### 7) Prediction Issues

- Ensure task ↔ weights match (e.g., `yolo26n-seg.pt` for segmentation).
- Use `classes`, `conf`, `imgsz` for filtering/quality:

```sh
yolo predict model=yolo26n.pt source=path/or/url imgsz=640 conf=0.25 classes=0,2 show=False save=True
```

- If blank outputs: lower `conf`, verify source path and image channels, try a canonical test image.

### 8) Export Issues

Start with ONNX; reduce complexity if needed, avoid adjusting `opset` unless necessary:

```sh
yolo export model=yolo26n.pt format=onnx imgsz=640 dynamic=False
```

If failure persists:

- Update `ultralytics`, `onnx`, `onnxruntime`.
- Try Python API export:

    ```python
    from ultralytics import YOLO

    YOLO("yolo26n.pt").export(format="onnx", imgsz=640)
    ```

### 9) Environment Modes

- macOS: CPU by default; use Metal/Accelerate or external GPU not supported by CUDA.
- Linux servers: prefer headless build and/or Docker for reproducibility.
- Docker quick start (GPU):
    ```sh
    t=ultralytics/ultralytics:latest
    sudo docker pull $t
    sudo docker run -it --ipc=host --runtime=nvidia --name ultralytics --gpus all $t
    sudo docker exec -it ultralytics yolo checks
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
