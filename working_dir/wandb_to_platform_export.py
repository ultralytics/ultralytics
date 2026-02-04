"""Export all wandb runs from a project to Ultralytics Platform."""

import os
import re

import requests
import wandb


def slugify(text):
    """Convert text to URL-safe slug (keeps underscores)."""
    if not text:
        return text
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9\s_-]", "", str(text).lower()).replace(" ", "-")).strip("-")[:128]


def export_run(run, platform_project: str, url: str, headers: dict):
    """Export a single wandb run to Platform."""
    config = run.config
    name = slugify(run.name)

    print(f"\n{'='*60}")
    print(f"Exporting: {run.name}")
    print(f"To: https://platform.ultralytics.com/{platform_project}/{name}")

    # 1. training_started
    resp = requests.post(url, headers=headers, json={
        "event": "training_started",
        "project": platform_project,
        "name": name,
        "data": {
            "trainArgs": {k: str(v) for k, v in config.items()},
            "epochs": config.get("epochs", 0),
            "device": str(config.get("device", "")),
        },
    })

    if not resp.ok:
        print(f"   FAILED: {resp.status_code} - {resp.text}")
        return

    model_id = resp.json().get("modelId")
    print(f"   Started (modelId: {model_id})")

    # 2. epoch_end
    history = list(run.scan_history())
    print(f"   Sending {len(history)} epochs...", end=" ")

    for i, row in enumerate(history):
        metrics = {k: v for k, v in row.items() if isinstance(v, (int, float)) and not k.startswith("_")}
        requests.post(url, headers=headers, json={
            "event": "epoch_end",
            "project": platform_project,
            "name": name,
            "modelId": model_id,
            "data": {"epoch": i, "metrics": metrics},
        })
    print("OK")

    # 3. training_complete
    summary = {k: v for k, v in run.summary.items() if isinstance(v, (int, float)) and not k.startswith("_")}
    requests.post(url, headers=headers, json={
        "event": "training_complete",
        "project": platform_project,
        "name": name,
        "modelId": model_id,
        "data": {"results": {"metrics": summary}},
    })
    print(f"   Completed!")


def export_project(wandb_project: str, platform_project: str, api_key: str = None, limit: int = None):
    """
    Export runs from a wandb project to Ultralytics Platform.

    Args:
        wandb_project: wandb project path (entity/project)
        platform_project: Platform project name (e.g. "esat/my-project")
        api_key: Ultralytics API key (or set ULTRALYTICS_API_KEY env var)
        limit: Max number of runs to export (latest first). None = all runs.
    """
    api_key = api_key or os.getenv("ULTRALYTICS_API_KEY")
    url = "https://platform.ultralytics.com/api/webhooks/training/metrics"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Get runs from wandb project (already sorted by latest first)
    api = wandb.Api()
    runs = list(api.runs(wandb_project))

    if limit:
        runs = runs[-limit:]  # latest runs are at the end

    print(f"Found {len(runs)} runs to export (limit={limit})")
    print(f"Exporting to Platform project: {platform_project}")

    for run in runs:
        try:
            export_run(run, platform_project, url, headers)
        except Exception as e:
            print(f"   ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Done! View at: https://platform.ultralytics.com/{platform_project}")


if __name__ == "__main__":
    export_project(
        wandb_project="esat/detr_trainings",  # entity/project
        platform_project="detr_trainings",  # your platform project
        limit=60,  # only latest 60 runs
    )
