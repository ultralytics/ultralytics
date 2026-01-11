# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def _slurm_master_addr():
    """Resolve the master address from SLURM nodelist if available."""
    nodelist = (
        os.environ.get("SLURM_NODELIST")
        or os.environ.get("SLURM_JOB_NODELIST")
        or os.environ.get("SLURM_STEP_NODELIST")
    )
    if not nodelist:
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        host = result.stdout.strip().splitlines()[0] if result.stdout else ""
        return host or None
    except Exception:
        pass
    # Fallback parser for common patterns like "node[01-03,05]"
    import re

    first = nodelist.split(",")[0]
    match = re.match(r"(?P<prefix>[A-Za-z0-9._-]+)\[(?P<range>[^\]]+)\]", first)
    if match:
        prefix = match.group("prefix")
        first_range = match.group("range").split(",")[0]
        num_match = re.match(r"\d+", first_range)
        if num_match:
            return f"{prefix}{num_match.group(0)}"
        return None
    return first if "[" not in first and "]" not in first else None


def _slurm_node_rank():
    """Resolve the node rank from SLURM if available."""
    node_rank = os.environ.get("SLURM_NODEID")
    if node_rank is None:
        return None
    try:
        return str(int(node_rank))
    except ValueError:
        return None


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

    nnodes = trainer.nnodes

    if not trainer.resume:
        try:
            shutil.rmtree(trainer.save_dir)  # remove the save_dir
        except FileNotFoundError:
            pass
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"

    master_port = os.environ.get("MASTER_PORT")
    if nnodes > 1:
        master_addr = os.environ.get("MASTER_ADDR")
        if master_addr and ("[" in master_addr or "]" in master_addr):
            master_addr = None
        node_rank = os.environ.get("NODE_RANK")
        if master_addr is None or node_rank is None:
            slurm_master_addr = _slurm_master_addr()
            slurm_node_rank = _slurm_node_rank()
            if master_addr is None:
                master_addr = slurm_master_addr
            if node_rank is None:
                node_rank = slurm_node_rank
        if master_addr is None or node_rank is None:
            raise ValueError(
                "MASTER_ADDR and NODE_RANK must be set for multi-node DDP "
                "(or available via SLURM_NODELIST/SLURM_NODEID)."
            )
        if master_port is None:
            master_port = "29500"
    else:
        master_addr = None
        node_rank = None
        if master_port is None:
            master_port = str(find_free_network_port())

    local_world_size = trainer.local_world_size
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nnodes",
        f"{nnodes}",
        "--nproc_per_node",
        f"{local_world_size}",
    ]
    if nnodes > 1:
        cmd.extend(["--node_rank", f"{node_rank}", "--master_addr", f"{master_addr}"])
    cmd.extend(["--master_port", f"{master_port}", file])
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
