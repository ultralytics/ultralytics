# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Build and smoke-validate export test virtual environments from the exporter registry."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from ultralytics.engine.exporter import EXPORT_ENVS


def isolated_env_ids():
    """Return export environments that should run outside the shared base CI environment."""
    return [env for env, recipe in EXPORT_ENVS.items() if recipe["python"]]


def build_env(env_id, root):
    """Build one export environment and run its smoke export commands."""
    recipe = EXPORT_ENVS[env_id]
    venv = root / env_id
    repo = Path(__file__).resolve().parents[2]

    shutil.rmtree(venv, ignore_errors=True)
    subprocess.run(["uv", "venv", str(venv), "--python", recipe["python"]], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    package = f"{repo}[{','.join(recipe['extras'])}]"
    indexes = [token for flag, url in recipe["indexes"] for token in (flag, url)]
    torch = [f"torch{recipe['torch']}"] if recipe["torch"] else []
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            package,
            "pytest",
            *torch,
            *recipe["requirements"],
            *indexes,
            "--index-strategy",
            "unsafe-best-match",
        ],
        check=True,
    )

    if recipe["env"]:
        site_packages = next(venv.glob("lib/python*/site-packages"))
        site_packages.joinpath("sitecustomize.py").write_text(
            "\n".join(f'import os; os.environ.setdefault("{k}", "{v}")' for k, v in recipe["env"].items()) + "\n"
        )

    env = {**os.environ, "YOLO_AUTOINSTALL": "false", **recipe["env"]}
    yolo = venv / ("Scripts/yolo.exe" if os.name == "nt" else "bin/yolo")
    for command in recipe["smoke"]:
        cmd = command.split()
        executable = yolo if cmd[0] == "yolo" else venv / "bin" / cmd[0]
        subprocess.run([str(executable), *cmd[1:]], check=True, cwd=venv, env=env)
    subprocess.run(["bash", "-c", f"rm -rf {venv}/yolo11n* {venv}/runs"], check=False)


def main():
    """Parse arguments and build requested export environments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", action="append", choices=isolated_env_ids())
    parser.add_argument("--list", action="store_true", help="Print isolated export environment ids and exit.")
    args = parser.parse_args()
    envs = args.env or isolated_env_ids()

    if args.list:
        print("\n".join(envs))
        return

    root = Path(os.environ.get("ULTRALYTICS_ISOLATED_VENVS", "/opt/venvs"))
    root.mkdir(parents=True, exist_ok=True)
    for env_id in envs:
        build_env(env_id, root)


if __name__ == "__main__":
    main()
