# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from __future__ import annotations

import os
import shutil
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


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


def generate_ddp_file(trainer):
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

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
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


def generate_ddp_command(trainer):
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


def generate_distributed_validation_file(validator):
    """Generate a temporary helper file to run distributed (multi-process) validation.

    This creates a small Python script under ``USER_CONFIG_DIR/DDP`` which imports the validator class, instantiates it
    with the validator's `args` (passed as ``overrides``), and runs the offline validation entrypoint. This is intended
    to be executed with ``torch.distributed.run`` (or the legacy ``torch.distributed.launch``) to spawn one process per
    GPU for parallel validation/inference. Note: this is for inference/validation only — it does not perform
    DistributedDataParallel (DDP) model wrapping or gradient synchronization.

    Args:
        validator: An instance of the project's validator (must have an ``args`` attribute).

    Returns:
        str: Path to the generated temporary Python file.
    """
    # Derive import path for the validator class
    module, name = f"{validator.__class__.__module__}.{validator.__class__.__name__}".rsplit(".", 1)

    content = f"""
# Ultralytics Multi-GPU validation temp file (auto-generated)
overrides = {vars(validator.args)}

if __name__ == "__main__":
    from {module} import {name}

    validator = {name}(args=overrides)
    # run the validator's offline validation routine
    validator.do_offline_validation()
"""

    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(validator)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_distributed_validation_command(validator):
    """Generate the command list to run distributed (multi-process) validation.

    Args:
        validator: An instance which exposes a ``world_size`` attribute (number of processes/GPU).

    Returns:
        tuple[list[str], str]: (command args list, path to the generated temp file)
    """
    # Calculate world size
    world_size = validator.world_size
    if world_size <= 1:
        raise ValueError("Distributed validation requires world_size > 1")

    file = generate_distributed_validation_file(validator)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(trainer, file):
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


def decide_world_size(device: str | tuple) -> int:
    import torch

    if isinstance(device, str) and len(device):  # i.e. device='0' or device='0,1,2,3'
        world_size = len(device.split(","))
    elif isinstance(device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
        world_size = len(device)
    elif device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
        world_size = 0
    elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
        world_size = 1  # default to device 0
    else:  # i.e. device=None or device=''
        world_size = 0

    return world_size
