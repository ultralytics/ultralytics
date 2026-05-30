#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build and smoke-validate every isolated export venv defined in EXPORT_ENVS.

Each venv installs the working tree (`uv pip install -e .[export-base]`) on its required Python and torch pin, then
runs its smoke export, which autoinstalls the vendor SDK via that backend's own check_requirements. The smoke export
IS the validation: check=True means an unbuildable or broken env fails the whole job loudly rather than silently
skipping. Adding a new isolated format needs only a new EXPORT_ENVS entry, never an edit here or in CI YAML.

The venv root defaults to /opt/venvs and is overridable with ULTRALYTICS_ISOLATED_VENVS (use a writable runner path,
e.g. $RUNNER_TEMP/venvs). Venvs are always rebuilt from scratch so a cached wheel layer can never serve stale code.
"""

import os
import shutil
import subprocess
from pathlib import Path

INDEX = ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match"]


def main():
    """Build and smoke-validate every isolated export venv defined in EXPORT_ENVS."""
    from ultralytics.engine.exporter import EXPORT_ENVS

    root = Path(os.environ.get("ULTRALYTICS_ISOLATED_VENVS", "/opt/venvs"))
    repo = Path(__file__).resolve().parents[2]
    for env, recipe in EXPORT_ENVS.items():
        venv = root / env
        shutil.rmtree(venv, ignore_errors=True)  # rebuild against the current working tree, no stale code
        subprocess.run(["uv", "venv", str(venv), "--python", recipe["python"]], check=True)
        torch = [f"torch{recipe['torch']}"] if recipe["torch"] else []
        subprocess.run(
            ["uv", "pip", "install", "--python", str(venv / "bin" / "python"),
             "-e", f"{repo}[export-base]", *torch, *recipe["requirements"], *INDEX],
            check=True,
        )  # fmt: skip
        cmd = recipe["smoke"].split()  # e.g. "yolo export format=imx model=yolo11n.pt imgsz=32"
        subprocess.run([str(venv / "bin" / cmd[0]), *cmd[1:]], check=True, cwd=venv)  # smoke export == validation
        subprocess.run(["bash", "-c", f"rm -rf {venv}/yolo11n* {venv}/runs"], check=False)  # tidy build artifacts
    print(f"Built and smoke-validated {len(EXPORT_ENVS)} isolated export venvs under {root}")


if __name__ == "__main__":
    main()
