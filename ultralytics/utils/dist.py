# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


def find_free_network_port() -> int:
    """Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer: BaseTrainer) -> str:
    """Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs. The file
    contains the necessary configuration to initialize the trainer in a distributed environment. Custom callbacks are
    serialized via pickle so they survive the subprocess boundary; callbacks that cannot be pickled (lambdas,
    closures, functions defined in `__main__`) are dropped with a warning (issue #6168).

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing training configuration and arguments.
            Must have args attribute and be a class instance.

    Returns:
        (str): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
        - Trainer class import
        - Configuration overrides from the trainer arguments
        - Model path configuration
        - Custom callbacks (pickled alongside the temp file)
        - Training initialization code
    """
    from . import callbacks as callback_utils

    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # Serialize augmentations to JSON-safe dicts to avoid NameError in DDP subprocess
    overrides = vars(trainer.args).copy()
    if overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.to_dict(t) for t in overrides["augmentations"]]

    # Serialize callbacks for DDP subprocess — skip non-picklable ones (issue #6168)
    callbacks_file = _serialize_callbacks(trainer, callback_utils.get_default_callbacks())

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
from pathlib import Path, PosixPath  # For model arguments stored as Path instead of str
overrides = {overrides}
callbacks_file = {callbacks_file!r}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT
    import pickle

    # Deserialize augmentations from dicts back to Albumentations transform objects
    if overrides.get("augmentations") is not None:
        import albumentations as A
        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]

    # Load custom callbacks pickled by the parent process (#6168)
    _callbacks = None
    if callbacks_file:
        with open(callbacks_file, "rb") as f:
            _callbacks = pickle.load(f)

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def _serialize_callbacks(trainer: BaseTrainer, default_cbs: dict) -> str | None:
    """Pickle serializable callbacks for DDP subprocess, dropping non-picklable ones with a warning.

    Args:
        trainer (BaseTrainer): Trainer whose callbacks to serialize.
        default_cbs (dict): Fresh default callbacks to identify which are user-added.

    Returns:
        (str | None): Path to the pickle file, or None if no user-added callbacks are serializable.
    """
    import pickle

    from . import LOGGER

    default_funcs = {f for funcs in default_cbs.values() for f in funcs}
    serializable, dropped = {}, []
    for event, funcs in trainer.callbacks.items():
        for func in funcs:
            if func in default_funcs:
                continue  # defaults are re-created in subprocess
            try:
                pickle.dumps(func)
                if getattr(func, "__module__", "") == "__main__":
                    raise pickle.PicklingError("defined in __main__")
                serializable.setdefault(event, []).append(func)
            except Exception:
                dropped.append(f"{event}={getattr(func, '__name__', repr(func))}")
    if dropped:
        LOGGER.warning(
            f"DDP info: {len(dropped)} custom callback(s) cannot be serialized for multi-GPU training and will be "
            f"dropped: {dropped}. Define callbacks in an importable module (not __main__ or a notebook) to preserve "
            "them across DDP."
        )
    if not serializable:
        return None
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_callbacks_", suffix=f"{id(trainer)}.pkl", mode="wb", dir=USER_CONFIG_DIR / "DDP", delete=False
    ) as f:
        pickle.dump(serializable, f)
    return f.name


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """Generate command for distributed training.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing configuration for distributed training.

    Returns:
        cmd (list[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(trainer: BaseTrainer, file: str) -> None:
    """Delete temporary file if created during distributed data parallel (DDP) training.

    This function checks if the provided file contains the trainer's ID in its name, indicating it was created as a
    temporary file for DDP training, and deletes it if so. Also removes the associated callbacks pickle file.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer used for distributed training.
        file (str): Path to the file that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
        # Remove callbacks pickle file (same id suffix, different prefix)
        for pkl in Path(file).parent.glob(f"_callbacks_*{id(trainer)}.pkl"):
            os.remove(pkl)
