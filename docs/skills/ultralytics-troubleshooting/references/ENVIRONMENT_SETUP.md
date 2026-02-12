# Environment Setup & Installation Guide

This guide helps diagnose and resolve environment setup issues when `yolo checks` fails or when ultralytics is not yet installed.

## Prerequisites Check

Before installing ultralytics, verify your system meets the basic requirements.

### 1. Detect Operating System

Determine your OS to inform subsequent steps:

```sh
# macOS/Linux:
uname -s

# Windows CMD/PowerShell:
ver
```

**Keep record of detected OS** - this affects installation commands and troubleshooting steps.

### 2. Check Python Availability

Verify Python 3.8+ is available:

```sh
python -V  # Check Python version

# If python command not found or version < 3.8, try:
python3 -V
```

**CRITICAL**: Python 3.8 or higher is required. If no suitable Python version is found, proceed to step 3.

### 3. Check for uv (Optional but Recommended)

`uv` is a fast Python package manager that can also manage/install Python:

```sh
uv --version
```

### 4. Decision Point

**If no Python 3.8+ and no uv:**
- **STOP HERE.** Guide the user to install:
  - **Option A (Recommended)**: Install `uv` from https://docs.astral.sh/uv/getting-started/installation/ - it can manage both Python and packages
  - **Option B**: Install Python 3.8+ (preference for 3.11 or 3.12) from https://www.python.org/downloads/
- After installation, return to step 2 above.

**IMPORTANT**: It is mandatory to document or store a memory for the following information before proceeding:
- Is `uv` available? 
    - If yes, all `uv` commands will be used.
- If not using `uv`, which Python command (`python` or `python3`) works? 
    - Necessary for future steps.
- If not using `uv`, What Python version has been installed?
    - If using `uv`, Python version will be specified in later step.

**If Python 3.8+ is available:**
- Proceed to [Virtual Environment Setup](#virtual-environment-setup) below.

**If `uv` is available but Python is not available:**
- Proceed to [Virtual Environment Setup](#virtual-environment-setup) below (`uv` can handle everything).

---

## Virtual Environment Setup

**Why virtual environments?**
Virtual environments isolate project dependencies and avoid conflicts with system packages. **mandatory** for `ultralytics` user installations.

### Check Current Environment

First, verify if you're already in a virtual environment:

```sh
python -c "import sys; print('Virtual env:', hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))"
```

If output shows `Virtual env: True`, you're already in a venv. Skip to [Installation Methods](#installation-methods).

If `Virtual env: False`, create and activate one:

### Creating a Virtual Environment

**Using uv (if available):**

```sh
uv venv .venv
# Optionally specify Python version:
uv venv .venv --python 3.11
```

**Using Python's built-in venv:**

```sh
python -m venv .venv
# or if python3 is your command:
python3 -m venv .venv
```

### Activating the Virtual Environment

Activate the environment before installing packages:

**macOS/Linux:**
```sh
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

After activation, your prompt should change to show `(.venv)`.

**Verify activation:**
```sh
which python  # macOS/Linux
# or
where python  # Windows
```

The path should point to your `.venv` directory.

---

## Installation Methods

Choose an installation method based on your needs and package manager preference.

### Verify Package Manager Preference

If not already determined in a previous step, ask the user which package manager they prefer:
- **uv** - Fast, modern (recommended if installed)
- **pip** - Default Python package manager (most common)
- **conda** - Good for bundled GPU dependencies

If user don't know or have no preference, use **pip** (included with Python).

**IMPORTANT**: It is mandatory to document or store a memory about what package manager is being used.

### Installation Options

Note: `uv` shares the same interface as `pip`. A command like `pip install -U ultralytics` would be `uv pip install -U ultralytics` for `uv`. There should be a memory or document that you can access for determining which package manager to use, check that before proceeding. All commands will be shown using `uv` since it's preferred, but if using `pip` for package management, then drop the preceding `uv` in the command.

#### Option 1: Stable Release

```sh
uv pip install ultralytics --upgrade
```

#### Option 2: Latest Development Version

Install from GitHub main branch:

```sh
uv pip install "git+https://github.com/ultralytics/ultralytics.git@main"
```

Use when you need bleeding-edge features or bug fixes not yet released.

#### Option 3: Editable Install (Development)

For working directly on the `ultralytics` codebase (assume cloned to local path):

```sh
# Navigate to the ultralytics repository root
cd ultralytics/

# Install in editable mode
uv pip install -e ".[dev]"
```

**Only use editable install when:**
- Actively developing ultralytics itself
- Testing local changes to the library
- Contributing to the ultralytics repository

#### Option 4: Headless (Servers without Display)

For servers or containers without GUI/display capabilities:

```sh
uv pip install ultralytics-opencv-headless
```

This avoids `libGL` and display-related errors. **Not recommended for local development** unless the user understands the implications of choosing this route.

#### Option 5: Conda (GPU Bundle)

Conda can install PyTorch with CUDA dependencies in one command. **Only use if conda is explicitly preferred:**

**IMPORTANT**: PyTorch>2.5.1 is NOT available using `conda`

```sh
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

**NOTE**: Adjust `pytorch-cuda` version to match the CUDA installation. If unclear, try running `nvidia-smi` and check the output for the CUDA version.

### Post-Installation Verification

After installation, verify `ultralytics` is available:

```sh
# Check CLI command
yolo version

# Check Python import
python -c "import ultralytics; print(ultralytics.__version__)"
```

Both should complete without errors and show the version number. Next run the system checks utility to ensure `ultralytics` is ready to use:

```sh
yolo checks
```

---

## Troubleshooting Common Issues

**REMINDER**: Check memories and/or your documentation about which package manager to use, as well as which Python command to use. All commands below shown using `python` and `uv`, but they should be adjusted as appropriate.

### Issue 1: "Command not found: yolo"

**Cause:** The installation directory is not in your PATH, or you're not in the venv where ultralytics was installed.

**Solutions:**
1. Verify virtual environment is activated (see "Activating the Virtual Environment" above)
2. Try running with `python -m`:
    ```sh
    python -m ultralytics version
    ```
3. Reinstall in the active environment:
    ```sh
    uv pip install --force-reinstall ultralytics
    ```

### Issue 2: "ModuleNotFoundError: No module named 'ultralytics'"

**Cause:** Ultralytics not installed in the current Python environment, or wrong Python interpreter is being used.

**Solutions:**
1. Verify you're in the correct virtual environment
2. Check which Python is active:
    ```sh
    which python  # macOS/Linux
    where python  # Windows
    ```
3. Install ultralytics in the active environment:
    ```sh
    uv pip install ultralytics
    ```

### Issue 3: Permission Errors During Installation

**Cause:** Trying to install to system Python without admin rights.

**Solutions:**
1. **Recommended:** Use a virtual environment (see "Virtual Environment Setup" above)
2. Install with `--user` flag (not recommended):
    ```sh
    uv pip install --user ultralytics
    ```

### Issue 4: Dependency Conflicts

**Cause:** Existing packages conflict with ultralytics requirements.

**Solutions:**
1. Create a fresh virtual environment:
    ```sh
    deactivate  # if in existing venv

    uv venv .venv_fresh --python 3.11  # when using uv
    python -m venv .venv_fresh  # otherwise use builtin venv

    source .venv_fresh/bin/activate
    
    uv pip install --no-cache -U ultralytics  # uv only, don't use cache
    pip install -U ultralytics
    ```
2. If the issue persists, upgrade `pip` and `setuptools`:
    ```sh
    uv pip install --upgrade pip setuptools wheel
    uv pip install ultralytics
    ```

### Issue 5: Network/Offline Installation

**Cause:** No internet access or firewall blocking PyPI.

**Solutions:**
1. Download packages on a connected machine (modify for current OS), notice command is not the same for `uv` and `pip` for download:
    - For `uv`
    ```sh
    uv run \ 
        --isolated \ 
        --no-sync \ 
        --with "pip" \ 
        python -m pip download ultralytics -d /path/to/packages
    ```
    - For `pip`
    ```sh
    pip download ultralytics -d /path/to/packages
    ```
2. Transfer packages directory to offline machine and install:
    ```sh
    uv pip install --no-index --find-links=/path/to/packages ultralytics
    ```

### Issue 6: torch/torchvision Installation Issues on macOS

**Cause:** ARM64 (Apple Silicon) requires specific builds.

**Solutions:**
1. Let pip install the correct versions automatically:
    ```sh
    uv pip install -U ultralytics
    ```
2. If issues persist, install PyTorch first:
    ```sh
    uv pip install -U torch torchvision
    uv pip install -U ultralytics
    ```

---

## Return to Main Workflow

Once installation is complete and verification succeeds, **return to SKILL.md** to continue with the full preflight checks and troubleshooting workflow.
