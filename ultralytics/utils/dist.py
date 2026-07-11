# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import inspect
import os
import re
import shutil
import sys
import tempfile
import textwrap
from typing import TYPE_CHECKING

from . import LOGGER, USER_CONFIG_DIR
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


def _get_custom_callback_injection_code(trainer: BaseTrainer) -> str:
    """Generate Python source code to re-register custom callbacks in DDP child processes.

    When DDP training uses a temporary Python file, user-registered callbacks added via
    ``model.add_callback()`` are not included because the temp file creates a fresh trainer.
    This function extracts the source code of user-defined callbacks and generates code to
    re-define and re-register them in each DDP child process.

    Only callbacks whose function objects are NOT in the built-in ``default_callbacks`` are
    serialized. Callbacks that fail ``inspect.getsource`` (lambdas, dynamic functions) are
    skipped with a warning.

    Args:
        trainer (BaseTrainer): The trainer instance whose callbacks should be serialized.

    Returns:
        (str): Indented Python source code to inject after the trainer is created in the DDP
        temp file, or an empty string if there are no custom callbacks to serialize.
    """
    from ultralytics.utils.callbacks.base import default_callbacks as _base_defaults

    # Collect built-in callback function objects so we can exclude them
    builtins: set = set()
    for cb_list in _base_defaults.values():
        for cb in cb_list:
            builtins.add(cb)

    # Walk trainer.callbacks and collect user-registered callbacks not in builtins
    custom_entries: list[tuple[str, str, str]] = []  # (event, func_name, dedented_source)
    for event, cb_list in trainer.callbacks.items():
        for cb in cb_list:
            if cb in builtins:
                continue
            try:
                source = inspect.getsource(cb)
                name: str | None = getattr(cb, "__name__", None)
                if not name or name == "<lambda>":
                    LOGGER.warning(
                        "WARNING ⚠️ Cannot serialize anonymous or lambda callback "
                        f"'{getattr(cb, '__name__', repr(cb))}' for DDP training - skipping"
                    )
                    continue
            except (OSError, TypeError):
                LOGGER.warning(
                    "WARNING ⚠️ Cannot serialize callback "
                    f"'{getattr(cb, '__name__', repr(cb))}' for DDP training - skipping. "
                    "Lambda and dynamically generated callbacks are not supported."
                )
                continue
            # Closures capture variables from enclosing scope that don't exist in DDP child
            if getattr(cb, "__closure__", None) is not None:
                LOGGER.warning(
                    "WARNING ⚠️ Cannot serialize closure callback "
                    f"'{name}' for DDP training — captured variables would be undefined - skipping"
                )
                continue
            # Detect callbacks that reference external imports that won't exist in DDP child.
            # __globals__ contains the module's global namespace; filter to non-builtin names.
            import builtins as _builtins_module
            _cb_globals = getattr(cb, "__globals__", {})
            _builtin_names = set(dir(_builtins_module))
            _unresolved = set()
            for _gname, _gval in _cb_globals.items():
                if _gname.startswith("_") or _gname in _builtin_names:
                    continue
                if _gname not in source:
                    continue
                if callable(_gval) or isinstance(_gval, type):
                    _unresolved.add(_gname)
            if _unresolved:
                LOGGER.warning(
                    "WARNING ⚠️ Callback '{}' references external names {} from module '{}' — "
                    "these must be importable in the DDP child process. The injection will "
                    "proceed, but the callback will raise NameError if the imports are missing.".format(
                        name, sorted(_unresolved), getattr(cb, "__module__", "unknown"))
                )
            # Normalize indentation so the source fits inside the ``if __name__ == "__main__":`` block
            source = textwrap.dedent(source)
            # Skip if the dedented source does not look like a function/async-function definition
            source_stripped = source.lstrip()
            if not (source_stripped.startswith(("def ", "async def ", "@"))):
                LOGGER.warning(
                    "WARNING ⚠️ Cannot serialize callback "
                    f"'{name}' for DDP training — source is not a valid function definition - skipping"
                )
                continue
            custom_entries.append((event, name, source))

    if not custom_entries:
        return ""

    lines: list[str] = ["", "    # === Injected custom callbacks (auto-generated for DDP) ==="]
    for i, (event, name, source) in enumerate(custom_entries):
        # Give each callback a unique local variable name to prevent collisions
        cb_var = f"__custom_cb_{i}"
        # Rename the function definition in-place so the unique name is used
        source = re.sub(rf"\bdef\s+{re.escape(name)}\b", f"def {cb_var}", source, count=1)
        indented = textwrap.indent(source, "    ")
        lines.append(indented)
        lines.append(f'    trainer.add_callback("{event}", {cb_var})')
    lines.append("    # === End injected custom callbacks ===\n")

    return "\n".join(lines)


def generate_ddp_file(trainer: BaseTrainer) -> str:
    """Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs. The file
    contains the necessary configuration to initialize the trainer in a distributed environment.

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
        - Training initialization code
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # Serialize augmentations to JSON-safe dicts to avoid NameError in DDP subprocess
    overrides = vars(trainer.args).copy()
    if overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.to_dict(t) for t in overrides["augmentations"]]

    # Collect user-registered custom callbacks so they survive the DDP subprocess launch
    custom_cb_code = _get_custom_callback_injection_code(trainer)

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
from pathlib import Path, PosixPath  # For model arguments stored as Path instead of str
overrides = {overrides}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    # Deserialize augmentations from dicts back to Albumentations transform objects
    if overrides.get("augmentations") is not None:
        import albumentations as A
        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
{custom_cb_code}
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
    temporary file for DDP training, and deletes it if so.

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
