---
description: Explore Ultralytics' utilities for distributed training including DDP file generation, command setup, and cleanup. Improve multi-node training efficiency.
keywords: Ultralytics, distributed training, DDP, multi-node training, network port, DDP file generation, DDP command, training utilities
---

# Reference for `ultralytics/utils/dist.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`find_free_network_port`](#ultralytics.utils.dist.find_free_network_port)
        - [`generate_ddp_file`](#ultralytics.utils.dist.generate_ddp_file)
        - [`generate_ddp_command`](#ultralytics.utils.dist.generate_ddp_command)
        - [`ddp_cleanup`](#ultralytics.utils.dist.ddp_cleanup)


## Function `ultralytics.utils.dist.find_free_network_port` {#ultralytics.utils.dist.find\_free\_network\_port}

```python
def find_free_network_port() -> int
```

Find a free port on localhost.

It is useful in single-node training when we don't want to connect to a real main node but have to set the `MASTER_PORT` environment variable.

**Returns**

| Type | Description |
| --- | --- |
| `int` | The available network port number. |

<details>
<summary>Source code in <code>ultralytics/utils/dist.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py#L18-L31"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.dist.generate_ddp_file` {#ultralytics.utils.dist.generate\_ddp\_file}

```python
def generate_ddp_file(trainer: BaseTrainer) -> str
```

Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

This function creates a temporary Python file that enables distributed training across multiple GPUs. The file contains the necessary configuration to initialize the trainer in a distributed environment.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The trainer containing training configuration and arguments.<br>    Must have args attribute and be a class instance. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `str` | Path to the generated temporary DDP file. |

!!! note "Notes"

    The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
    - Trainer class import
    - Configuration overrides from the trainer arguments
    - Model path configuration
    - Training initialization code

<details>
<summary>Source code in <code>ultralytics/utils/dist.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py#L34-L81"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
from pathlib import Path, PosixPath  # For model arguments stored as Path instead of str
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.dist.generate_ddp_command` {#ultralytics.utils.dist.generate\_ddp\_command}

```python
def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]
```

Generate command for distributed training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The trainer containing configuration for distributed training. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `cmd (list[str])` | The command to execute for distributed training. |
| `file (str)` | Path to the temporary file created for DDP training. |

<details>
<summary>Source code in <code>ultralytics/utils/dist.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py#L84-L111"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.dist.ddp_cleanup` {#ultralytics.utils.dist.ddp\_cleanup}

```python
def ddp_cleanup(trainer: BaseTrainer, file: str) -> None
```

Delete temporary file if created during distributed data parallel (DDP) training.

This function checks if the provided file contains the trainer's ID in its name, indicating it was created as a temporary file for DDP training, and deletes it if so.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The trainer used for distributed training. | *required* |
| `file` | `str` | Path to the file that might need to be deleted. | *required* |

**Examples**

```python
>>> trainer = YOLOTrainer()
>>> file = "/tmp/ddp_temp_123456789.py"
>>> ddp_cleanup(trainer, file)
```

<details>
<summary>Source code in <code>ultralytics/utils/dist.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/dist.py#L114-L130"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br><br>
