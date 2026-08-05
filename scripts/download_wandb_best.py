# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Download best.pt from a wandb run and rename it as <project>-<run-name>.pt.

Usage:
    .venv/bin/python scripts/download_wandb_best.py n-deep-muon-0.5-sgd-0.5-150
    .venv/bin/python scripts/download_wandb_best.py n-deep-muon-0.5-sgd-0.5-150 --out weights/my-model.pt
"""

import argparse
import shutil
from pathlib import Path

import wandb


def download_best(run_name: str, project: str, entity: str, out_file: Path | None) -> Path:
    """Download the 'best' model artifact of a run, default name <project>-<run-name>.pt."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    if not runs:
        raise SystemExit(f"No run named '{run_name}' in {entity}/{project}")
    run = runs[0]
    print(f"Found run: {run.name} (id: {run.id}, state: {run.state})")

    art = api.artifact(f"{entity}/{project}/run_{run.id}_model:best")
    src = Path(art.get_entry("best.pt").download(root="/tmp/wandb_dl"))

    dst = out_file or Path("weights") / f"{project}-{run.name}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), dst)
    print(f"Saved: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_name", help="wandb run display name, e.g. n-deep-muon-0.5-sgd-0.5-150")
    p.add_argument("--project", default="YOLO27-obj365")
    p.add_argument("--entity", default="laughing")
    p.add_argument("--out", type=Path, default=None, help="output file path (default: weights/<project>-<run>.pt)")
    args = p.parse_args()
    download_best(args.run_name, args.project, args.entity, args.out)
