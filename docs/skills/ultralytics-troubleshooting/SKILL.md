---
name: ultralytics-troubleshooting
description: Diagnose and resolve common Ultralytics YOLO issues. Use when users have problems with installation (imports fail, CLI not found, CUDA issues), training (slow/NaN loss, CPU-only, bad metrics, dataset errors), prediction (no/empty results, wrong classes), or export (ONNX/TensorRT/OpenVINO failures). Covers environment setup, GPU alignment, and basic workflow validation.
license: AGPL-3.0
metadata:
    author: Burhan-Q
    version: "1.0"
    ultralytics-version: ">=8.4.0"
---

# Ultralytics Troubleshooting – Agent Skill

Quickly diagnose and resolve common Ultralytics YOLO issues across install, runtime, training, prediction, export, and environment setup. See [References](#references-docs) at the end for deeper guidance.

## Preconditions

- Python 3.8+ available (preferably 3.11+).
- Internet access for package install/update (unless using local editable install).
- Python virtual environment (mandatory, unless user acknowledges acceptance of risks when using global install).
- If GPU expected: correct CUDA + PyTorch build alignment.

## Safety / Side Effects

- Verify with user before modifying files, installing packages, or changing environment.
- Prefer non-destructive diagnostics (e.g., `yolo checks`) before reinstalling or changing configs.
- Installing/upgrading packages may change environment behavior.
- Avoid deleting user data; create new files under runs/ or temp dirs.

---

## Quick Decision Tree

1. Using containerized environment (Docker)?
    - Yes → Use Docker image and commands; skip to Step 3 for GPU checks.
    - No → Proceed to Step 2.
2. Is Python virtual environment in use?
    - No → Advise user to create and activate a virtual environment.
    - Yes → Ensure virtual environment is active and package is installed.
3. Can `yolo version` or `python -c "import ultralytics as u; print(u.__version__)"` run?
    - No → Ask user to activate the correct environment or install package in current environment.
4. Expect GPU? Run `yolo checks` or try `python -c "import torch; print(torch.cuda.is_available())"`.
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

**Basic first step, check installation and system information:**

Run built-in diagnostics, first time call will take time to process, reports package installation information and system details:

```sh
yolo checks
```

If this succeeds, then ask for additional clarification regarding the user's issue(s) that they need help troubleshooting.

**If `yolo checks` fails or `ultralytics` is not installed:**

The system environment may not be properly configured. Advise user on proper environment setup:

- Python installation, Python 3.8+ required (preferably 3.11+)
- Virtual environment (uv, venv, conda, etc.) mandatory for isolation unless user understands risks of global install.
- GPU users: ensure compatible CUDA + PyTorch versions.

After completing environment setup with user, attempt to run `yolo checks` again to verify installation before proceeding.

Interpretation:

- If no GPU found, but GPU expected → go to GPU Fix (Step 3).
- If packages are outdated or show problem (denoted with ❌) → go to Install/Repair (Step 2).
- Repeated failures → direct user to Ultralytics community support ([Discord][UltraDiscord], [Forums][UltraForums], [Reddit][UltraReddit], or [GitHub][UltraGitHub]). Guide the user to create a [Minimum Reproducible Example][MRE] before posting — this significantly improves response quality from the community.

### 2) Install / Repair

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
    python -c "import torch; print('cuda:', torch.cuda.is_available()); print('cap:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'N/A'); print('cudnn:', torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A')"
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
- For custom training, try pre-trained weights first to isolate data vs. optimization issues.

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

- Ensure task ↔ weights match. Model naming: `yolo26<SIZE>[-TASK].pt` where SIZE is `n/s/m/l/x`:
    - No suffix (`.pt`) → detection only | `-seg` → segmentation | `-pose` → pose | `-cls` → classification | `-obb` → oriented bbox
    - Custom trained models default to `best.pt` — check model output to confirm task.
- Key prediction args (see [CLI docs][CLIDocs] for full reference):
    - `conf` — confidence threshold (default 0.25); lower for more results
    - `source` — file, folder, URL, or `0` for webcam; omit to use packaged test images
    - `classes` — filter to specific class IDs | `imgsz` — input size (default 640)
    - `save=True` — save results to `runs/detect/predict`
    - WARNING: large videos consume significant resources; start with a single image.

- Minimum command (uses packaged images, good for sanity check):

```sh
yolo predict
```

- With user-specified args:

```sh
yolo predict model=yolo26n.pt source=path/to/image.jpg imgsz=640 conf=0.25 save=True
```

- If blank outputs: lower `conf`, verify source path, check if target objects match model classes, try without `source` for packaged images.

### 8) Export Issues

Start with ONNX; reduce complexity if needed, avoid adjusting `opset` unless necessary:

```sh
yolo export model=yolo26n.pt format=onnx imgsz=640 dynamic=False
```

The `ultralytics` package should install necessary dependencies on demand for exports, if failure persists:

- Update `ultralytics`, `onnx`, `onnxruntime`.
- Try Python API export:

    ```python
    from ultralytics import YOLO

    YOLO("yolo26n.pt").export(format="onnx", imgsz=640)
    ```

### 9) Platform Notes

- macOS: Uses Metal GPU acceleration (ARM64) or CPU-only. No CUDA support.
- Linux: may need system packages (`libgl1`, `libglib2.0-0`, etc.). CUDA requires matching drivers + toolkit.
- Docker (useful for isolating environment issues, especially with GPU):
    ```sh
    t=ultralytics/ultralytics:latest
    sudo docker pull $t
    sudo docker run -it --ipc=host --runtime=nvidia --name ultralytics --gpus all $t
    sudo docker exec -it ultralytics yolo checks
    ```

### 10) Settings

View current Ultralytics configuration:

```sh
yolo settings
```

Shows default directories (`datasets_dir`, `weights_dir`, `runs_dir`) and other configuration. Useful when models or datasets download to unexpected locations. Usually not the source of user issues, but can help diagnose path-related problems.

---

## Minimal Python Sanity (Optional)

- Prediction

```python
from ultralytics import ASSETS, YOLO

model = YOLO("yolo26n.pt")
res = model.predict(ASSETS / "bus.jpg")
print(res[0].summary())
# Inspect output
```

- Training & Validation

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="coco8.yaml", epochs=1, imgsz=320)
# Inspect output
```

## What To Return To The User

- Environment summary from `yolo checks`, including any issues with installation/dependencies.
- The working command(s) used and where results are stored (e.g. path/to/runs/, using OS appropriate `/`, `\`).
- Any constraints or remaining gaps (e.g., GPU not present, export format limitations).

## References (Docs)

- Install & CLI basics:
    - Cloned repo: docs/en/quickstart.md
    - Web: https://docs.ultralytics.com/quickstart
- Common issues guide:
    - Cloned repo: docs/en/guides/yolo-common-issues.md
    - Web: https://docs.ultralytics.com/guides/yolo-common-issues
- CLI usage & args:
    - Cloned repo: docs/en/usage/cli.md
    - Web: https://docs.ultralytics.com/usage/cli
- Python usage:
    - Cloned repo: docs/en/usage/python.md
    - Web: https://docs.ultralytics.com/usage/python
- Docker quickstart:
    - Cloned repo: docs/en/guides/docker-quickstart.md
    - Web: https://docs.ultralytics.com/guides/docker-quickstart
- Conda quickstart:
    - Cloned repo: docs/en/guides/conda-quickstart.md
    - Web: https://docs.ultralytics.com/guides/conda-quickstart
- Datasets overview:
    - Cloned repo: docs/en/datasets/index.md
    - Web: https://docs.ultralytics.com/datasets/index
- Training tips:
    - Cloned repo: docs/en/guides/model-training-tips.md
    - Web: https://docs.ultralytics.com/guides/model-training-tips
- Minimum Reproducible Example:
    - Cloned repo: docs/en/help/minimum-reproducible-example.md
    - Web: https://docs.ultralytics.com/help/minimum-reproducible-example

[UltraDiscord]: https://ultralytics.com/discord
[UltraForums]: https://community.ultralytics.com
[UltraReddit]: https://www.reddit.com/r/ultralytics/
[UltraGitHub]: https://github.com/ultralytics/ultralytics
[MRE]: https://docs.ultralytics.com/help/minimum-reproducible-example
[CLIDocs]: https://docs.ultralytics.com/usage/cli
