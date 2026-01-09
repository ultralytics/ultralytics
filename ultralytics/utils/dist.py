# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import subprocess
import sys
import tempfile

from . import LOGGER, USER_CONFIG_DIR, colorstr
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


def generate_ddp_file(obj):
    """Generate a DDP (Distributed Data Parallel) file for multi-GPU training/validation.

    This function creates a temporary Python file that enables distributed training/validation across multiple GPUs. The
    file contains the necessary configuration to initialize the trainer/validator in a distributed environment.

    Args:
        obj (ultralytics.engine.trainer.BaseTrainer | ultralytics.engine.validator.BaseValidator): The trainer/validator
            containing training/validation configuration and arguments. Must have args attribute and be a
            class instance.

    Returns:
        (str): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
        - Trainer/validator class import
        - Configuration overrides from the trainer arguments
        - Model path configuration
        - Training/validation initialization code
    """
    module, name = f"{obj.__class__.__module__}.{obj.__class__.__name__}".rsplit(".", 1)
    content = f"""
# Ultralytics Multi-GPU training/validation temp file (should be automatically deleted after use)
overrides = {vars(obj.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.torch_utils import setup_ddp

    setup_ddp()  # initialize dist
    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
"""
    if hasattr(obj, "train"):
        content += f"""
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(obj.hub_session, "model_url", obj.args.model)}"
    results = trainer.train()
"""
    else:
        content += f"""
    validator = {name}(args=overrides)
    results = validator()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(obj)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(obj):
    """Generate command for distributed training/validation.

    Args:
        obj (ultralytics.engine.trainer.BaseTrainer | ultralytics.engine.validator.BaseValidator): The trainer/validator
            containing configuration for distributed training.

    Returns:
        cmd (list[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if hasattr(obj, "train") and not obj.resume:
        shutil.rmtree(obj.save_dir)  # remove the save_dir
    file = generate_ddp_file(obj)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{obj.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(obj, file):
    """Delete temporary file if created during distributed data parallel (DDP) training/validation.

    This function checks if the provided file contains the trainer's/validator's ID in its name, indicating it was
    created as a temporary file for DDP training, and deletes it if so.

    Args:
        obj (ultralytics.engine.trainer.BaseTrainer | ultralytics.engine.validator.BaseValidator): The trainer/validator
            used for distributed training/validation.
        file (str): Path to the file that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(obj)}.py" in file:  # if temp_file suffix in file
        os.remove(file)


def run_ddp(obj):
    """Run DDP training/validation in a subprocess."""
    cmd, file = generate_ddp_command(obj)
    try:
        LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except Exception as e:
        raise e
    finally:
        ddp_cleanup(obj, str(file))
