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
- Repeated failures → suggest user visit Ultralytics [Discord][UltraDiscord], [Forums][UltraForums], [Reddit][UltraReddit], or [GitHub][UltraGitHub] for community support.

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

- Ensure task ↔ weights match (e.g., `yolo26n-seg.pt` for segmentation).
    - NOTE: replace `<SIZE>` with the model size (n/s/m/l/x) as appropriate.
    - `yolo26<SIZE>.pt` is detection-only, will not work for segmentation or pose tasks.
    - `yolo26<SIZE>-seg.pt` is segmentation-capable, can be used for detection or segmentation.
    - `yolo26<SIZE>-pose.pt` is pose-capable, can be used for detection or pose, but not segmentation.
    - `yolo26<SIZE>-cls.pt` is classification-capable, can only be used for classification.
    - `yolo26<SIZE>-obb.pt` is oriented bounding box-capable, can only be used for OBB detection.
    - Custom training models may not include task indicator in filename, default filename for custom models are usually `best.pt`, but may be modified by users, so observe output of `yolo predict model=<MODEL>.pt` (replace `<MODEL>` with the appropriate model filename).
- Use args `classes`, `conf`, `imgsz` for filtering/quality:
    - `classes` limits prediction outputs to specific class IDs
    - `conf` changes confidence threshold for predictions, default is 0.25, lower for more results, higher for fewer.
    - `imgsz` can affect detection quality, default is 640
    - `source` can be a local file, folder, URL, or video stream (use `source=0` for webcam if available); verify path and format, if not given inference run on two packaged images.
        - WARNING: Large videos may consume significant system resources; start with an image first, then short video or small directory of images.
    - `show` will display results in a window, disable by default or if issues with displaying image(s).
    - `save` will save results to `runs/detect/predict` by default, verify output files for results.

- Minimum command expected to work with packaged images:
```sh
yolo predict
```

- Attempt with user specified model, image `source`, and other args as needed:
```sh
yolo predict model=yolo26n.pt imgsz=640 conf=0.25 classes=0,2 show=False save=True
```

- If blank outputs: lower `conf`, verify source path (if any), ask about objects to detect vs model classes, try without `source` argument for basic image.

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

### 9) Environment

- macOS: Uses Metal GPU acceleration (ARM64) or CPU-only. External GPUs not supported by CUDA.
- Linux: potential system packages missing, advise `sudo apt-get update && sudo apt-get install -y <missing-packages>`. Some systems attempting to use CUDA may require additional setup (drivers, toolkit, etc.).
- If available, Docker can be extremely useful to help isolate system variables (when GPU is available):
```sh
t=ultralytics/ultralytics:latest
sudo docker pull $t
sudo docker run -it --ipc=host --runtime=nvidia --name ultralytics --gpus all $t
sudo docker exec -it ultralytics yolo checks
```

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