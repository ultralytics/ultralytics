"""Central config for W&B project + local SSD / NFS mirror roots.

Why this module exists:
- Ultralytics' built-in W&B callback (ultralytics/utils/callbacks/wb.py:138) does
  ``project=str(trainer.args.project).replace("/", "-")``. Passing an absolute path as
  ``project=`` (e.g. ``/home/fatih/runs/yolo-next-encoder``) mangles the W&B project to
  ``-home-fatih-runs-yolo-next-encoder``. The fix is to pass a clean W&B project name as
  ``project=`` and an absolute local path as ``save_dir=``; Ultralytics' ``get_save_dir``
  (ultralytics/cfg/__init__.py:395) honors ``save_dir`` verbatim, bypassing the
  project+name join.
- Ultralytics' ``check_resume`` (ultralytics/engine/trainer.py:841) overwrites most
  caller-supplied args with the checkpoint's ``train_args`` (whitelist: imgsz, batch,
  device, close_mosaic, augmentations, save_period, workers, cache, patience, time,
  freeze, val, plots). For cross-machine or relocated resumes, project/name/save_dir
  must be patched on the checkpoint itself, not on the caller side.

Callers use ``run_paths(name)`` for fresh runs and ``patch_resume(ckpt)`` for resumes.
"""

from __future__ import annotations

from pathlib import Path

WANDB_ENTITY = "fca"
WANDB_PROJECT = "yolo-next-encoder"
LOCAL_ROOT = Path("/home/fatih/runs/yolo-next-encoder")
NFS_MIRROR_ROOT = Path("/data/shared-datasets/fatih-runs/classify/yolo-next-encoder")
SYNC_INTERVAL_SEC = 600

assert LOCAL_ROOT.is_absolute() and str(LOCAL_ROOT).startswith("/home/"), (
    f"LOCAL_ROOT must be absolute and under /home/ to decouple from NFS, got {LOCAL_ROOT}"
)


def run_paths(name: str, exist_ok: bool = False) -> dict:
    """Return W&B project + absolute local save_dir kwargs for ``model.train``.

    Args:
        name (str): Run name, used as W&B display name and ``save_dir`` leaf.
        exist_ok (bool, optional): Allow ``save_dir`` to already exist.

    Returns:
        (dict): Kwargs with ``project``, ``name``, ``save_dir``, ``exist_ok``.
    """
    return dict(project=WANDB_PROJECT, name=name, save_dir=str(LOCAL_ROOT / name), exist_ok=exist_ok)


def multi_results_csv(parent_name: str, root: Path = NFS_MIRROR_ROOT) -> Path:
    """Return the ``<root>/<parent_name>/multi_results.csv`` path for a multi_det parent run.

    Single source of truth for this layout so the multi_det writer and reader cannot drift on path or filename.

    Args:
        parent_name (str): multi_det parent run name.
        root (Path, optional): Base root, the shared NFS_MIRROR_ROOT or the host-local LOCAL_ROOT.

    Returns:
        (Path): Absolute path to the parent run's aggregate CSV.
    """
    return root / parent_name / "multi_results.csv"


def patch_resume(ckpt_path, name: str | None = None, device=None, data: str | None = None) -> str:
    """Rewrite a checkpoint's ``train_args`` to clean W&B project + absolute local save_dir, in place.

    Needed because Ultralytics' ``check_resume`` restores project/name/save_dir/data from the checkpoint, not caller
    kwargs; without this, a resume on a different machine or save_dir inherits whatever the original trainer baked in.

    Args:
        ckpt_path (str | Path): Checkpoint to patch (local or NFS path).
        name (str, optional): Override run name. Defaults to the checkpoint's existing name.
        device (int | str, optional): Override CUDA device (whitelisted for resume, e.g. when the new machine exposes
            the target physical GPU as a different CUDA index).
        data (str, optional): Override dataset path, e.g. when the resuming host mounts the dataset at a different
            location (``data`` is NOT in ``check_resume``'s override whitelist so it must be baked into the checkpoint).

    Returns:
        (str): Absolute path of the patched checkpoint (same as input, for chaining).
    """
    import torch

    ckpt_path = Path(ckpt_path).expanduser().resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.setdefault("train_args", {}) or {}
    run_name = name or train_args.get("name") or ckpt_path.parent.parent.name
    train_args.update(project=WANDB_PROJECT, name=run_name, save_dir=str(LOCAL_ROOT / run_name), exist_ok=True)
    if device is not None:
        train_args["device"] = device
    if data is not None:
        train_args["data"] = data
    ckpt["train_args"] = train_args
    torch.save(ckpt, ckpt_path)
    return str(ckpt_path)
