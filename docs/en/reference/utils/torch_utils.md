---
description: Explore valuable torch utilities from Ultralytics for optimized model performance, including device selection, model fusion, and inference optimization.
keywords: Ultralytics, torch utils, model optimization, device selection, inference optimization, model fusion, CPU info, PyTorch tools
---

# Reference for `ultralytics/utils/torch_utils.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ModelEMA`](#ultralytics.utils.torch_utils.ModelEMA)
        - [`EarlyStopping`](#ultralytics.utils.torch_utils.EarlyStopping)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ModelEMA.update`](#ultralytics.utils.torch_utils.ModelEMA.update)
        - [`ModelEMA.update_attr`](#ultralytics.utils.torch_utils.ModelEMA.update_attr)
        - [`EarlyStopping.__call__`](#ultralytics.utils.torch_utils.EarlyStopping.__call__)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`torch_distributed_zero_first`](#ultralytics.utils.torch_utils.torch_distributed_zero_first)
        - [`smart_inference_mode`](#ultralytics.utils.torch_utils.smart_inference_mode)
        - [`autocast`](#ultralytics.utils.torch_utils.autocast)
        - [`get_cpu_info`](#ultralytics.utils.torch_utils.get_cpu_info)
        - [`get_gpu_info`](#ultralytics.utils.torch_utils.get_gpu_info)
        - [`select_device`](#ultralytics.utils.torch_utils.select_device)
        - [`time_sync`](#ultralytics.utils.torch_utils.time_sync)
        - [`fuse_conv_and_bn`](#ultralytics.utils.torch_utils.fuse_conv_and_bn)
        - [`fuse_deconv_and_bn`](#ultralytics.utils.torch_utils.fuse_deconv_and_bn)
        - [`model_info`](#ultralytics.utils.torch_utils.model_info)
        - [`get_num_params`](#ultralytics.utils.torch_utils.get_num_params)
        - [`get_num_gradients`](#ultralytics.utils.torch_utils.get_num_gradients)
        - [`model_info_for_loggers`](#ultralytics.utils.torch_utils.model_info_for_loggers)
        - [`get_flops`](#ultralytics.utils.torch_utils.get_flops)
        - [`get_flops_with_torch_profiler`](#ultralytics.utils.torch_utils.get_flops_with_torch_profiler)
        - [`initialize_weights`](#ultralytics.utils.torch_utils.initialize_weights)
        - [`scale_img`](#ultralytics.utils.torch_utils.scale_img)
        - [`copy_attr`](#ultralytics.utils.torch_utils.copy_attr)
        - [`intersect_dicts`](#ultralytics.utils.torch_utils.intersect_dicts)
        - [`is_parallel`](#ultralytics.utils.torch_utils.is_parallel)
        - [`unwrap_model`](#ultralytics.utils.torch_utils.unwrap_model)
        - [`one_cycle`](#ultralytics.utils.torch_utils.one_cycle)
        - [`init_seeds`](#ultralytics.utils.torch_utils.init_seeds)
        - [`unset_deterministic`](#ultralytics.utils.torch_utils.unset_deterministic)
        - [`strip_optimizer`](#ultralytics.utils.torch_utils.strip_optimizer)
        - [`convert_optimizer_state_dict_to_fp16`](#ultralytics.utils.torch_utils.convert_optimizer_state_dict_to_fp16)
        - [`cuda_memory_usage`](#ultralytics.utils.torch_utils.cuda_memory_usage)
        - [`profile_ops`](#ultralytics.utils.torch_utils.profile_ops)
        - [`attempt_compile`](#ultralytics.utils.torch_utils.attempt_compile)


## Class `ultralytics.utils.torch_utils.ModelEMA` {#ultralytics.utils.torch\_utils.ModelEMA}

```python
ModelEMA(self, model, decay = 0.9999, tau = 2000, updates = 0)
```

Updated Exponential Moving Average (EMA) implementation.

Keeps a moving average of everything in the model state_dict (parameters and buffers). For EMA details see References.

To disable EMA set the `enabled` attribute to `False`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Model to create EMA for. | *required* |
| `decay` | `float, optional` | Maximum EMA decay rate. | `0.9999` |
| `tau` | `int, optional` | EMA decay time constant. | `2000` |
| `updates` | `int, optional` | Initial number of updates. | `0` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `ema` | `nn.Module` | Copy of the model in evaluation mode. |
| `updates` | `int` | Number of EMA updates. |
| `decay` | `function` | Decay function that determines the EMA weight. |
| `enabled` | `bool` | Whether EMA is enabled. |
| `References` |  |  |
| `- https://github.com/rwightman/pytorch-image-models` |  |  |
| `- https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage` |  |  |

**Methods**

| Name | Description |
| --- | --- |
| [`update`](#ultralytics.utils.torch_utils.ModelEMA.update) | Update EMA parameters. |
| [`update_attr`](#ultralytics.utils.torch_utils.ModelEMA.update_attr) | Copy attributes from model to EMA, with options to include/exclude certain attributes. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L609-L670"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ModelEMA:
    """Updated Exponential Moving Average (EMA) implementation.

    Keeps a moving average of everything in the model state_dict (parameters and buffers). For EMA details see
    References.

    To disable EMA set the `enabled` attribute to `False`.

    Attributes:
        ema (nn.Module): Copy of the model in evaluation mode.
        updates (int): Number of EMA updates.
        decay (function): Decay function that determines the EMA weight.
        enabled (bool): Whether EMA is enabled.

    References:
        - https://github.com/rwightman/pytorch-image-models
        - https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        """Initialize EMA for 'model' with given arguments.

        Args:
            model (nn.Module): Model to create EMA for.
            decay (float, optional): Maximum EMA decay rate.
            tau (int, optional): EMA decay time constant.
            updates (int, optional): Initial number of updates.
        """
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True
```
</details>

<br>

### Method `ultralytics.utils.torch_utils.ModelEMA.update` {#ultralytics.utils.torch\_utils.ModelEMA.update}

```python
def update(self, model)
```

Update EMA parameters.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Model to update EMA from. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L644-L658"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update(self, model):
    """Update EMA parameters.

    Args:
        model (nn.Module): Model to update EMA from.
    """
    if self.enabled:
        self.updates += 1
        d = self.decay(self.updates)

        msd = unwrap_model(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
```
</details>

<br>

### Method `ultralytics.utils.torch_utils.ModelEMA.update_attr` {#ultralytics.utils.torch\_utils.ModelEMA.update\_attr}

```python
def update_attr(self, model, include = (), exclude = ("process_group", "reducer"))
```

Copy attributes from model to EMA, with options to include/exclude certain attributes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Model to copy attributes from. | *required* |
| `include` | `tuple, optional` | Attributes to include. | `()` |
| `exclude` | `tuple, optional` | Attributes to exclude. | `("process_group", "reducer")` |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L661-L670"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def update_attr(self, model, include=(), exclude=("process_group", "reducer")):
    """Copy attributes from model to EMA, with options to include/exclude certain attributes.

    Args:
        model (nn.Module): Model to copy attributes from.
        include (tuple, optional): Attributes to include.
        exclude (tuple, optional): Attributes to exclude.
    """
    if self.enabled:
        copy_attr(self.ema, model, include, exclude)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.torch_utils.EarlyStopping` {#ultralytics.utils.torch\_utils.EarlyStopping}

```python
EarlyStopping(self, patience = 50)
```

Early stopping class that stops training when a specified number of epochs have passed without improvement.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `patience` | `int, optional` | Number of epochs to wait after fitness stops improving before stopping. | `50` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `best_fitness` | `float` | Best fitness value observed. |
| `best_epoch` | `int` | Epoch where best fitness was observed. |
| `patience` | `int` | Number of epochs to wait after fitness stops improving before stopping. |
| `possible_stop` | `bool` | Flag indicating if stopping may occur next epoch. |

**Methods**

| Name | Description |
| --- | --- |
| [`__call__`](#ultralytics.utils.torch_utils.EarlyStopping.__call__) | Check whether to stop training. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L860-L908"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class EarlyStopping:
    """Early stopping class that stops training when a specified number of epochs have passed without improvement.

    Attributes:
        best_fitness (float): Best fitness value observed.
        best_epoch (int): Epoch where best fitness was observed.
        patience (int): Number of epochs to wait after fitness stops improving before stopping.
        possible_stop (bool): Flag indicating if stopping may occur next epoch.
    """

    def __init__(self, patience=50):
        """Initialize early stopping object.

        Args:
            patience (int, optional): Number of epochs to wait after fitness stops improving before stopping.
        """
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float("inf")  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch
```
</details>

<br>

### Method `ultralytics.utils.torch_utils.EarlyStopping.__call__` {#ultralytics.utils.torch\_utils.EarlyStopping.\_\_call\_\_}

```python
def __call__(self, epoch, fitness)
```

Check whether to stop training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `epoch` | `int` | Current epoch of training. | *required* |
| `fitness` | `float` | Fitness value of current epoch. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if training should stop, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L881-L908"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __call__(self, epoch, fitness):
    """Check whether to stop training.

    Args:
        epoch (int): Current epoch of training.
        fitness (float): Fitness value of current epoch.

    Returns:
        (bool): True if training should stop, False otherwise.
    """
    if fitness is None:  # check if fitness=None (happens when val=False)
        return False

    if fitness > self.best_fitness or self.best_fitness == 0:  # allow for early zero-fitness stage of training
        self.best_epoch = epoch
        self.best_fitness = fitness
    delta = epoch - self.best_epoch  # epochs without improvement
    self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
    stop = delta >= self.patience  # stop training if patience exceeded
    if stop:
        prefix = colorstr("EarlyStopping: ")
        LOGGER.info(
            f"{prefix}Training stopped early as no improvement observed in last {self.patience} epochs. "
            f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
            f"To update EarlyStopping(patience={self.patience}) pass a new patience value, "
            f"i.e. `patience=300` or use `patience=0` to disable EarlyStopping."
        )
    return stop
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.torch_distributed_zero_first` {#ultralytics.utils.torch\_utils.torch\_distributed\_zero\_first}

```python
def torch_distributed_zero_first(local_rank: int)
```

Ensure all processes in distributed training wait for the local master (rank 0) to complete a task first.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `local_rank` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L62-L71"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """Ensure all processes in distributed training wait for the local master (rank 0) to complete a task first."""
    initialized = dist.is_available() and dist.is_initialized()
    use_ids = initialized and dist.get_backend() == "nccl"

    if initialized and local_rank not in {-1, 0}:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
    yield
    if initialized and local_rank == 0:
        dist.barrier(device_ids=[local_rank]) if use_ids else dist.barrier()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.smart_inference_mode` {#ultralytics.utils.torch\_utils.smart\_inference\_mode}

```python
def smart_inference_mode()
```

Apply torch.inference_mode() decorator if torch>=1.10.0, else torch.no_grad() decorator.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L74-L84"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def smart_inference_mode():
    """Apply torch.inference_mode() decorator if torch>=1.10.0, else torch.no_grad() decorator."""

    def decorate(fn):
        """Apply appropriate torch decorator for inference mode based on torch version."""
        if TORCH_1_9 and torch.is_inference_mode_enabled():
            return fn  # already in inference_mode, act as a pass-through
        else:
            return (torch.inference_mode if TORCH_1_10 else torch.no_grad)()(fn)

    return decorate
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.autocast` {#ultralytics.utils.torch\_utils.autocast}

```python
def autocast(enabled: bool, device: str = "cuda")
```

Get the appropriate autocast context manager based on PyTorch version and AMP setting.

This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `enabled` | `bool` | Whether to enable automatic mixed precision. | *required* |
| `device` | `str, optional` | The device to use for autocast. | `"cuda"` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.amp.autocast` | The appropriate autocast context manager. |

**Examples**

```python
>>> with autocast(enabled=True):
...     # Your mixed precision operations here
...     pass
```

!!! note "Notes"

    - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
    - For older versions, it uses `torch.cuda.amp.autocast`.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L87-L112"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def autocast(enabled: bool, device: str = "cuda"):
    """Get the appropriate autocast context manager based on PyTorch version and AMP setting.

    This function returns a context manager for automatic mixed precision (AMP) training that is compatible with both
    older and newer versions of PyTorch. It handles the differences in the autocast API between PyTorch versions.

    Args:
        enabled (bool): Whether to enable automatic mixed precision.
        device (str, optional): The device to use for autocast.

    Returns:
        (torch.amp.autocast): The appropriate autocast context manager.

    Examples:
        >>> with autocast(enabled=True):
        ...     # Your mixed precision operations here
        ...     pass

    Notes:
        - For PyTorch versions 1.13 and newer, it uses `torch.amp.autocast`.
        - For older versions, it uses `torch.cuda.amp.autocast`.
    """
    if TORCH_1_13:
        return torch.amp.autocast(device, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.get_cpu_info` {#ultralytics.utils.torch\_utils.get\_cpu\_info}

```python
def get_cpu_info()
```

Return a string with system CPU information, i.e. 'Apple M2'.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L116-L125"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@functools.lru_cache
def get_cpu_info():
    """Return a string with system CPU information, i.e. 'Apple M2'."""
    from ultralytics.utils import PERSISTENT_CACHE  # avoid circular import error

    if "cpu_info" not in PERSISTENT_CACHE:
        try:
            PERSISTENT_CACHE["cpu_info"] = CPUInfo.name()
        except Exception:
            pass
    return PERSISTENT_CACHE.get("cpu_info", "unknown")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.get_gpu_info` {#ultralytics.utils.torch\_utils.get\_gpu\_info}

```python
def get_gpu_info(index)
```

Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `index` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L129-L132"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@functools.lru_cache
def get_gpu_info(index):
    """Return a string with system GPU information, i.e. 'Tesla T4, 15102MiB'."""
    properties = torch.cuda.get_device_properties(index)
    return f"{properties.name}, {properties.total_memory / (1 << 20):.0f}MiB"
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.select_device` {#ultralytics.utils.torch\_utils.select\_device}

```python
def select_device(device = "", newline = False, verbose = True)
```

Select the appropriate PyTorch device based on the provided arguments.

The function takes a string specifying the device or a torch.device object and returns a torch.device object representing the selected device. The function also validates the number of available devices and raises an exception if the requested device(s) are not available.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `device` | `str | torch.device, optional` | Device string or torch.device object. Options include 'cpu', 'cuda', '0',<br>    '0,1,2,3', 'mps', or '-1' for auto-select. Defaults to auto-selecting the first available GPU, or CPU if no<br>    GPU is available. | `""` |
| `newline` | `bool, optional` | If True, adds a newline at the end of the log string. | `False` |
| `verbose` | `bool, optional` | If True, logs the device information. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.device` | Selected device. |

**Examples**

```python
>>> select_device("cuda:0")
device(type='cuda', index=0)

>>> select_device("cpu")
device(type='cpu')
```

!!! note "Notes"

    Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L135-L229"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def select_device(device="", newline=False, verbose=True):
    """Select the appropriate PyTorch device based on the provided arguments.

    The function takes a string specifying the device or a torch.device object and returns a torch.device object
    representing the selected device. The function also validates the number of available devices and raises an
    exception if the requested device(s) are not available.

    Args:
        device (str | torch.device, optional): Device string or torch.device object. Options include 'cpu', 'cuda', '0',
            '0,1,2,3', 'mps', or '-1' for auto-select. Defaults to auto-selecting the first available GPU, or CPU if no
            GPU is available.
        newline (bool, optional): If True, adds a newline at the end of the log string.
        verbose (bool, optional): If True, logs the device information.

    Returns:
        (torch.device): Selected device.

    Examples:
        >>> select_device("cuda:0")
        device(type='cuda', index=0)

        >>> select_device("cpu")
        device(type='cpu')

    Notes:
        Sets the 'CUDA_VISIBLE_DEVICES' environment variable for specifying which GPUs to use.
    """
    if isinstance(device, torch.device) or str(device).startswith(("tpu", "intel", "vulkan")):
        return device

    s = f"Ultralytics {__version__} 🚀 Python-{PYTHON_VERSION} torch-{TORCH_VERSION} "
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'

    # Auto-select GPUs
    if "-1" in device:
        from ultralytics.utils.autodevice import GPUInfo

        # Replace each -1 with a selected GPU or remove it
        parts = device.split(",")
        selected = GPUInfo().select_idle_gpu(count=parts.count("-1"), min_memory_fraction=0.2)
        for i in range(len(parts)):
            if parts[i] == "-1":
                parts[i] = str(selected.pop(0)) if selected else ""
        device = ",".join(p for p in parts if p)

    cpu = device == "cpu"
    mps = device in {"mps", "mps:0"}  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])  # remove sequential commas, i.e. "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            LOGGER.info(s)
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # i.e. "0,1" -> ["0", "1"]
        space = " " * len(s)
        for i, d in enumerate(devices):
            s += f"{'' if i == 0 else space}CUDA:{d} ({get_gpu_info(i)})\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and TORCH_2_0 and torch.backends.mps.is_available():
        # Prefer MPS if available
        s += f"MPS ({get_cpu_info()})\n"
        arg = "mps"
    else:  # revert to CPU
        s += f"CPU ({get_cpu_info()})\n"
        arg = "cpu"

    if arg in {"cpu", "mps"}:
        torch.set_num_threads(NUM_THREADS)  # reset OMP_NUM_THREADS for cpu training
    if verbose:
        LOGGER.info(s if newline else s.rstrip())
    return torch.device(arg)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.time_sync` {#ultralytics.utils.torch\_utils.time\_sync}

```python
def time_sync()
```

Return PyTorch-accurate time.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L232-L236"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def time_sync():
    """Return PyTorch-accurate time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.fuse_conv_and_bn` {#ultralytics.utils.torch\_utils.fuse\_conv\_and\_bn}

```python
def fuse_conv_and_bn(conv, bn)
```

Fuse Conv2d and BatchNorm2d layers for inference optimization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `conv` | `nn.Conv2d` | Convolutional layer to fuse. | *required* |
| `bn` | `nn.BatchNorm2d` | Batch normalization layer to fuse. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `nn.Conv2d` | The fused convolutional layer with gradients disabled. |

**Examples**

```python
>>> conv = nn.Conv2d(3, 16, 3)
>>> bn = nn.BatchNorm2d(16)
>>> fused_conv = fuse_conv_and_bn(conv, bn)
```

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L239-L269"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d and BatchNorm2d layers for inference optimization.

    Args:
        conv (nn.Conv2d): Convolutional layer to fuse.
        bn (nn.BatchNorm2d): Batch normalization layer to fuse.

    Returns:
        (nn.Conv2d): The fused convolutional layer with gradients disabled.

    Examples:
        >>> conv = nn.Conv2d(3, 16, 3)
        >>> bn = nn.BatchNorm2d(16)
        >>> fused_conv = fuse_conv_and_bn(conv, bn)
    """
    # Compute fused weights
    w_conv = conv.weight.view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    conv.weight.data = torch.mm(w_bn, w_conv).view(conv.weight.shape)

    # Compute fused bias
    b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    if conv.bias is None:
        conv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        conv.bias.data = fused_bias

    return conv.requires_grad_(False)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.fuse_deconv_and_bn` {#ultralytics.utils.torch\_utils.fuse\_deconv\_and\_bn}

```python
def fuse_deconv_and_bn(deconv, bn)
```

Fuse ConvTranspose2d and BatchNorm2d layers for inference optimization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `deconv` | `nn.ConvTranspose2d` | Transposed convolutional layer to fuse. | *required* |
| `bn` | `nn.BatchNorm2d` | Batch normalization layer to fuse. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `nn.ConvTranspose2d` | The fused transposed convolutional layer with gradients disabled. |

**Examples**

```python
>>> deconv = nn.ConvTranspose2d(16, 3, 3)
>>> bn = nn.BatchNorm2d(3)
>>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
```

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L272-L302"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d and BatchNorm2d layers for inference optimization.

    Args:
        deconv (nn.ConvTranspose2d): Transposed convolutional layer to fuse.
        bn (nn.BatchNorm2d): Batch normalization layer to fuse.

    Returns:
        (nn.ConvTranspose2d): The fused transposed convolutional layer with gradients disabled.

    Examples:
        >>> deconv = nn.ConvTranspose2d(16, 3, 3)
        >>> bn = nn.BatchNorm2d(3)
        >>> fused_deconv = fuse_deconv_and_bn(deconv, bn)
    """
    # Compute fused weights
    w_deconv = deconv.weight.view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    deconv.weight.data = torch.mm(w_bn, w_deconv).view(deconv.weight.shape)

    # Compute fused bias
    b_conv = torch.zeros(deconv.out_channels, device=deconv.weight.device) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_bias = torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn

    if deconv.bias is None:
        deconv.register_parameter("bias", nn.Parameter(fused_bias))
    else:
        deconv.bias.data = fused_bias

    return deconv.requires_grad_(False)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.model_info` {#ultralytics.utils.torch\_utils.model\_info}

```python
def model_info(model, detailed = False, verbose = True, imgsz = 640)
```

Print and return detailed model information layer by layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Model to analyze. | *required* |
| `detailed` | `bool, optional` | Whether to print detailed layer information. | `False` |
| `verbose` | `bool, optional` | Whether to print model information. | `True` |
| `imgsz` | `int | list, optional` | Input image size. | `640` |

**Returns**

| Type | Description |
| --- | --- |
| `tuple` | Tuple containing: |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L305-L347"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def model_info(model, detailed=False, verbose=True, imgsz=640):
    """Print and return detailed model information layer by layer.

    Args:
        model (nn.Module): Model to analyze.
        detailed (bool, optional): Whether to print detailed layer information.
        verbose (bool, optional): Whether to print model information.
        imgsz (int | list, optional): Input image size.

    Returns:
        (tuple): Tuple containing:
            - n_l (int): Number of layers.
            - n_p (int): Number of parameters.
            - n_g (int): Number of gradients.
            - flops (float): GFLOPs.
    """
    if not verbose:
        return
    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    layers = __import__("collections").OrderedDict((n, m) for n, m in model.named_modules() if len(m._modules) == 0)
    n_l = len(layers)  # number of layers
    if detailed:
        h = f"{'layer':>5}{'name':>40}{'type':>20}{'gradient':>10}{'parameters':>12}{'shape':>20}{'mu':>10}{'sigma':>10}"
        LOGGER.info(h)
        for i, (mn, m) in enumerate(layers.items()):
            mn = mn.replace("module_list.", "")
            mt = m.__class__.__name__
            if len(m._parameters):
                for pn, p in m.named_parameters():
                    LOGGER.info(
                        f"{i:>5g}{f'{mn}.{pn}':>40}{mt:>20}{p.requires_grad!r:>10}{p.numel():>12g}{list(p.shape)!s:>20}{p.mean():>10.3g}{p.std():>10.3g}{str(p.dtype).replace('torch.', ''):>15}"
                    )
            else:  # layers with no learnable params
                LOGGER.info(f"{i:>5g}{mn:>40}{mt:>20}{False!r:>10}{0:>12g}{[]!s:>20}{'-':>10}{'-':>10}{'-':>15}")

    flops = get_flops(model, imgsz)  # imgsz may be int or list, i.e. imgsz=640 or imgsz=[640, 320]
    fused = " (fused)" if getattr(model, "is_fused", lambda: False)() else ""
    fs = f", {flops:.1f} GFLOPs" if flops else ""
    yaml_file = getattr(model, "yaml_file", "") or getattr(model, "yaml", {}).get("yaml_file", "")
    model_name = Path(yaml_file).stem.replace("yolo", "YOLO") or "Model"
    LOGGER.info(f"{model_name} summary{fused}: {n_l:,} layers, {n_p:,} parameters, {n_g:,} gradients{fs}")
    return n_l, n_p, n_g, flops
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.get_num_params` {#ultralytics.utils.torch\_utils.get\_num\_params}

```python
def get_num_params(model)
```

Return the total number of parameters in a YOLO model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L350-L352"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_num_params(model):
    """Return the total number of parameters in a YOLO model."""
    return sum(x.numel() for x in model.parameters())
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.get_num_gradients` {#ultralytics.utils.torch\_utils.get\_num\_gradients}

```python
def get_num_gradients(model)
```

Return the total number of parameters with gradients in a YOLO model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L355-L357"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_num_gradients(model):
    """Return the total number of parameters with gradients in a YOLO model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.model_info_for_loggers` {#ultralytics.utils.torch\_utils.model\_info\_for\_loggers}

```python
def model_info_for_loggers(trainer)
```

Return model info dict with useful model information.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The trainer object containing model and validation data. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary containing model parameters, GFLOPs, and inference speeds. |

**Examples**

```python
YOLOv8n info for loggers
>>> results = {
...    "model/parameters": 3151904,
...    "model/GFLOPs": 8.746,
...    "model/speed_ONNX(ms)": 41.244,
...    "model/speed_TensorRT(ms)": 3.211,
...    "model/speed_PyTorch(ms)": 18.755,
...}
```

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L360-L390"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def model_info_for_loggers(trainer):
    """Return model info dict with useful model information.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer object containing model and validation data.

    Returns:
        (dict): Dictionary containing model parameters, GFLOPs, and inference speeds.

    Examples:
        YOLOv8n info for loggers
        >>> results = {
        ...    "model/parameters": 3151904,
        ...    "model/GFLOPs": 8.746,
        ...    "model/speed_ONNX(ms)": 41.244,
        ...    "model/speed_TensorRT(ms)": 3.211,
        ...    "model/speed_PyTorch(ms)": 18.755,
        ...}
    """
    if trainer.args.profile:  # profile ONNX and TensorRT times
        from ultralytics.utils.benchmarks import ProfileModels

        results = ProfileModels([trainer.last], device=trainer.device).run()[0]
        results.pop("model/name")
    else:  # only return PyTorch times from most recent validation
        results = {
            "model/parameters": get_num_params(trainer.model),
            "model/GFLOPs": round(get_flops(trainer.model), 3),
        }
    results["model/speed_PyTorch(ms)"] = round(trainer.validator.speed["inference"], 3)
    return results
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.get_flops` {#ultralytics.utils.torch\_utils.get\_flops}

```python
def get_flops(model, imgsz = 640)
```

Calculate FLOPs (floating point operations) for a model in GFLOPs.

Attempts two calculation methods: first with a stride-based tensor for efficiency, then falls back to full image size if needed (e.g., for RTDETR models). Returns 0.0 if thop library is unavailable or calculation fails.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | The model to calculate FLOPs for. | *required* |
| `imgsz` | `int | list, optional` | Input image size. | `640` |

**Returns**

| Type | Description |
| --- | --- |
| `float` | The model's GFLOPs (billions of floating point operations). |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L393-L430"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_flops(model, imgsz=640):
    """Calculate FLOPs (floating point operations) for a model in GFLOPs.

    Attempts two calculation methods: first with a stride-based tensor for efficiency, then falls back to full image
    size if needed (e.g., for RTDETR models). Returns 0.0 if thop library is unavailable or calculation fails.

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model's GFLOPs (billions of floating point operations).
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    if not thop:
        return 0.0  # if not installed return 0.0 GFLOPs

    try:
        model = unwrap_model(model)
        p = next(model.parameters())
        if not isinstance(imgsz, list):
            imgsz = [imgsz, imgsz]  # expand if int/float
        try:
            # Method 1: Use stride-based input tensor
            stride = max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32  # max stride
            im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
            flops = thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # stride GFLOPs
            return flops * imgsz[0] / stride * imgsz[1] / stride  # imgsz GFLOPs
        except Exception:
            # Method 2: Use actual image size (required for RTDETR models)
            im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
            return thop.profile(deepcopy(model), inputs=[im], verbose=False)[0] / 1e9 * 2  # imgsz GFLOPs
    except Exception:
        return 0.0
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.get_flops_with_torch_profiler` {#ultralytics.utils.torch\_utils.get\_flops\_with\_torch\_profiler}

```python
def get_flops_with_torch_profiler(model, imgsz = 640)
```

Compute model FLOPs using torch profiler (alternative to thop package, but 2-10x slower).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | The model to calculate FLOPs for. | *required* |
| `imgsz` | `int | list, optional` | Input image size. | `640` |

**Returns**

| Type | Description |
| --- | --- |
| `float` | The model's GFLOPs (billions of floating point operations). |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L433-L463"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_flops_with_torch_profiler(model, imgsz=640):
    """Compute model FLOPs using torch profiler (alternative to thop package, but 2-10x slower).

    Args:
        model (nn.Module): The model to calculate FLOPs for.
        imgsz (int | list, optional): Input image size.

    Returns:
        (float): The model's GFLOPs (billions of floating point operations).
    """
    if not TORCH_2_0:  # torch profiler implemented in torch>=2.0
        return 0.0
    model = unwrap_model(model)
    p = next(model.parameters())
    if not isinstance(imgsz, list):
        imgsz = [imgsz, imgsz]  # expand if int/float
    try:
        # Use stride size for input tensor
        stride = (max(int(model.stride.max()), 32) if hasattr(model, "stride") else 32) * 2  # max stride
        im = torch.empty((1, p.shape[1], stride, stride), device=p.device)  # input image in BCHW format
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
        flops = flops * imgsz[0] / stride * imgsz[1] / stride  # 640x640 GFLOPs
    except Exception:
        # Use actual image size for input tensor (i.e. required for RTDETR models)
        im = torch.empty((1, p.shape[1], *imgsz), device=p.device)  # input image in BCHW format
        with torch.profiler.profile(with_flops=True) as prof:
            model(im)
        flops = sum(x.flops for x in prof.key_averages()) / 1e9
    return flops
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.initialize_weights` {#ultralytics.utils.torch\_utils.initialize\_weights}

```python
def initialize_weights(model)
```

Initialize model weights, biases, and module settings to default values.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L466-L476"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def initialize_weights(model):
    """Initialize model weights, biases, and module settings to default values."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.scale_img` {#ultralytics.utils.torch\_utils.scale\_img}

```python
def scale_img(img, ratio = 1.0, same_shape = False, gs = 32)
```

Scale and pad an image tensor, optionally maintaining aspect ratio and padding to gs multiple.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img` | `torch.Tensor` | Input image tensor. | *required* |
| `ratio` | `float, optional` | Scaling ratio. | `1.0` |
| `same_shape` | `bool, optional` | Whether to maintain the same shape. | `False` |
| `gs` | `int, optional` | Grid size for padding. | `32` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Scaled and padded image tensor. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L479-L498"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    """Scale and pad an image tensor, optionally maintaining aspect ratio and padding to gs multiple.

    Args:
        img (torch.Tensor): Input image tensor.
        ratio (float, optional): Scaling ratio.
        same_shape (bool, optional): Whether to maintain the same shape.
        gs (int, optional): Grid size for padding.

    Returns:
        (torch.Tensor): Scaled and padded image tensor.
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)  # resize
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.copy_attr` {#ultralytics.utils.torch\_utils.copy\_attr}

```python
def copy_attr(a, b, include = (), exclude = ())
```

Copy attributes from object 'b' to object 'a', with options to include/exclude certain attributes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `a` | `Any` | Destination object to copy attributes to. | *required* |
| `b` | `Any` | Source object to copy attributes from. | *required* |
| `include` | `tuple, optional` | Attributes to include. If empty, all attributes are included. | `()` |
| `exclude` | `tuple, optional` | Attributes to exclude. | `()` |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L501-L514"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def copy_attr(a, b, include=(), exclude=()):
    """Copy attributes from object 'b' to object 'a', with options to include/exclude certain attributes.

    Args:
        a (Any): Destination object to copy attributes to.
        b (Any): Source object to copy attributes from.
        include (tuple, optional): Attributes to include. If empty, all attributes are included.
        exclude (tuple, optional): Attributes to exclude.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.intersect_dicts` {#ultralytics.utils.torch\_utils.intersect\_dicts}

```python
def intersect_dicts(da, db, exclude = ())
```

Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `da` | `dict` | First dictionary. | *required* |
| `db` | `dict` | Second dictionary. | *required* |
| `exclude` | `tuple, optional` | Keys to exclude. | `()` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary of intersecting keys with matching shapes. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L517-L528"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def intersect_dicts(da, db, exclude=()):
    """Return a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values.

    Args:
        da (dict): First dictionary.
        db (dict): Second dictionary.
        exclude (tuple, optional): Keys to exclude.

    Returns:
        (dict): Dictionary of intersecting keys with matching shapes.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.is_parallel` {#ultralytics.utils.torch\_utils.is\_parallel}

```python
def is_parallel(model)
```

Return True if model is of type DP or DDP.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | Model to check. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if model is DataParallel or DistributedDataParallel. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L531-L540"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def is_parallel(model):
    """Return True if model is of type DP or DDP.

    Args:
        model (nn.Module): Model to check.

    Returns:
        (bool): True if model is DataParallel or DistributedDataParallel.
    """
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.unwrap_model` {#ultralytics.utils.torch\_utils.unwrap\_model}

```python
def unwrap_model(m: nn.Module) -> nn.Module
```

Unwrap compiled and parallel models to get the base model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `m` | `nn.Module` | A model that may be wrapped by torch.compile (._orig_mod) or parallel wrappers such as<br>    DataParallel/DistributedDataParallel (.module). | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `nn.Module` | The unwrapped base model without compile or parallel wrappers. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L543-L559"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def unwrap_model(m: nn.Module) -> nn.Module:
    """Unwrap compiled and parallel models to get the base model.

    Args:
        m (nn.Module): A model that may be wrapped by torch.compile (._orig_mod) or parallel wrappers such as
            DataParallel/DistributedDataParallel (.module).

    Returns:
        (nn.Module): The unwrapped base model without compile or parallel wrappers.
    """
    while True:
        if hasattr(m, "_orig_mod") and isinstance(m._orig_mod, nn.Module):
            m = m._orig_mod
        elif hasattr(m, "module") and isinstance(m.module, nn.Module):
            m = m.module
        else:
            return m
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.one_cycle` {#ultralytics.utils.torch\_utils.one\_cycle}

```python
def one_cycle(y1 = 0.0, y2 = 1.0, steps = 100)
```

Return a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `y1` | `float, optional` | Initial value. | `0.0` |
| `y2` | `float, optional` | Final value. | `1.0` |
| `steps` | `int, optional` | Number of steps. | `100` |

**Returns**

| Type | Description |
| --- | --- |
| `function` | Lambda function for computing the sinusoidal ramp. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L562-L573"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def one_cycle(y1=0.0, y2=1.0, steps=100):
    """Return a lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf.

    Args:
        y1 (float, optional): Initial value.
        y2 (float, optional): Final value.
        steps (int, optional): Number of steps.

    Returns:
        (function): Lambda function for computing the sinusoidal ramp.
    """
    return lambda x: max((1 - math.cos(x * math.pi / steps)) / 2, 0) * (y2 - y1) + y1
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.init_seeds` {#ultralytics.utils.torch\_utils.init\_seeds}

```python
def init_seeds(seed = 0, deterministic = False)
```

Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `seed` | `int, optional` | Random seed. | `0` |
| `deterministic` | `bool, optional` | Whether to set deterministic algorithms. | `False` |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L576-L598"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def init_seeds(seed=0, deterministic=False):
    """Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html.

    Args:
        seed (int, optional): Random seed.
        deterministic (bool, optional): Whether to set deterministic algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic:
        if TORCH_2_0:
            torch.use_deterministic_algorithms(True, warn_only=True)  # warn if deterministic is not possible
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["PYTHONHASHSEED"] = str(seed)
        else:
            LOGGER.warning("Upgrade to torch>=2.0.0 for deterministic training.")
    else:
        unset_deterministic()
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.unset_deterministic` {#ultralytics.utils.torch\_utils.unset\_deterministic}

```python
def unset_deterministic()
```

Unset all the configurations applied for deterministic training.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L601-L606"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def unset_deterministic():
    """Unset all the configurations applied for deterministic training."""
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ.pop("PYTHONHASHSEED", None)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.strip_optimizer` {#ultralytics.utils.torch\_utils.strip\_optimizer}

```python
def strip_optimizer(f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None) -> dict[str, Any]
```

Strip optimizer from 'f' to finalize training, optionally save as 's'.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `f` | `str | Path` | File path to model to strip the optimizer from. | `"best.pt"` |
| `s` | `str, optional` | File path to save the model with stripped optimizer to. If not provided, 'f' will be<br>    overwritten. | `""` |
| `updates` | `dict, optional` | A dictionary of updates to overlay onto the checkpoint before saving. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | The combined checkpoint dictionary. |

**Examples**

```python
>>> from pathlib import Path
>>> from ultralytics.utils.torch_utils import strip_optimizer
>>> for f in Path("path/to/model/checkpoints").rglob("*.pt"):
...     strip_optimizer(f)
```

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L673-L730"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def strip_optimizer(f: str | Path = "best.pt", s: str = "", updates: dict[str, Any] | None = None) -> dict[str, Any]:
    """Strip optimizer from 'f' to finalize training, optionally save as 's'.

    Args:
        f (str | Path): File path to model to strip the optimizer from.
        s (str, optional): File path to save the model with stripped optimizer to. If not provided, 'f' will be
            overwritten.
        updates (dict, optional): A dictionary of updates to overlay onto the checkpoint before saving.

    Returns:
        (dict): The combined checkpoint dictionary.

    Examples:
        >>> from pathlib import Path
        >>> from ultralytics.utils.torch_utils import strip_optimizer
        >>> for f in Path("path/to/model/checkpoints").rglob("*.pt"):
        ...     strip_optimizer(f)
    """
    try:
        x = torch_load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"Skipping {f}, not a valid Ultralytics model: {e}")
        return {}

    metadata = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
        x["model"].criterion = None  # strip loss criterion
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates", "scaler":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']

    # Save
    combined = {**metadata, **x, **(updates or {})}
    torch.save(combined, s or f)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")
    return combined
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.convert_optimizer_state_dict_to_fp16` {#ultralytics.utils.torch\_utils.convert\_optimizer\_state\_dict\_to\_fp16}

```python
def convert_optimizer_state_dict_to_fp16(state_dict)
```

Convert the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `state_dict` | `dict` | Optimizer state dictionary. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Converted optimizer state dictionary with FP16 tensors. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L733-L747"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def convert_optimizer_state_dict_to_fp16(state_dict):
    """Convert the state_dict of a given optimizer to FP16, focusing on the 'state' key for tensor conversions.

    Args:
        state_dict (dict): Optimizer state dictionary.

    Returns:
        (dict): Converted optimizer state dictionary with FP16 tensors.
    """
    for state in state_dict["state"].values():
        for k, v in state.items():
            if k != "step" and isinstance(v, torch.Tensor) and v.dtype is torch.float32:
                state[k] = v.half()

    return state_dict
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.cuda_memory_usage` {#ultralytics.utils.torch\_utils.cuda\_memory\_usage}

```python
def cuda_memory_usage(device = None)
```

Monitor and manage CUDA memory usage.

This function checks if CUDA is available and, if so, empties the CUDA cache to free up unused memory. It then yields a dictionary containing memory usage information, which can be updated by the caller. Finally, it updates the dictionary with the amount of memory reserved by CUDA on the specified device.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `device` | `torch.device, optional` | The CUDA device to query memory usage for. | `None` |

**Yields**

| Type | Description |
| --- | --- |
| `dict` | A dictionary with a key 'memory' initialized to 0, which will be updated with the reserved memory. |

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L751-L772"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@contextmanager
def cuda_memory_usage(device=None):
    """Monitor and manage CUDA memory usage.

    This function checks if CUDA is available and, if so, empties the CUDA cache to free up unused memory. It then
    yields a dictionary containing memory usage information, which can be updated by the caller. Finally, it updates the
    dictionary with the amount of memory reserved by CUDA on the specified device.

    Args:
        device (torch.device, optional): The CUDA device to query memory usage for.

    Yields:
        (dict): A dictionary with a key 'memory' initialized to 0, which will be updated with the reserved memory.
    """
    cuda_info = dict(memory=0)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            yield cuda_info
        finally:
            cuda_info["memory"] = torch.cuda.memory_reserved(device)
    else:
        yield cuda_info
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.profile_ops` {#ultralytics.utils.torch\_utils.profile\_ops}

```python
def profile_ops(input, ops, n = 10, device = None, max_num_obj = 0)
```

Ultralytics speed, memory and FLOPs profiler.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `input` | `torch.Tensor | list` | Input tensor(s) to profile. | *required* |
| `ops` | `nn.Module | list` | Model or list of operations to profile. | *required* |
| `n` | `int, optional` | Number of iterations to average. | `10` |
| `device` | `str | torch.device, optional` | Device to profile on. | `None` |
| `max_num_obj` | `int, optional` | Maximum number of objects for simulation. | `0` |

**Returns**

| Type | Description |
| --- | --- |
| `list` | Profile results for each operation. |

**Examples**

```python
>>> from ultralytics.utils.torch_utils import profile_ops
>>> input = torch.randn(16, 3, 640, 640)
>>> m1 = lambda x: x * torch.sigmoid(x)
>>> m2 = nn.SiLU()
>>> profile_ops(input, [m1, m2], n=100)  # profile over 100 iterations
```

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L775-L857"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def profile_ops(input, ops, n=10, device=None, max_num_obj=0):
    """Ultralytics speed, memory and FLOPs profiler.

    Args:
        input (torch.Tensor | list): Input tensor(s) to profile.
        ops (nn.Module | list): Model or list of operations to profile.
        n (int, optional): Number of iterations to average.
        device (str | torch.device, optional): Device to profile on.
        max_num_obj (int, optional): Maximum number of objects for simulation.

    Returns:
        (list): Profile results for each operation.

    Examples:
        >>> from ultralytics.utils.torch_utils import profile_ops
        >>> input = torch.randn(16, 3, 640, 640)
        >>> m1 = lambda x: x * torch.sigmoid(x)
        >>> m2 = nn.SiLU()
        >>> profile_ops(input, [m1, m2], n=100)  # profile over 100 iterations
    """
    try:
        import thop
    except ImportError:
        thop = None  # conda support without 'ultralytics-thop' installed

    results = []
    if not isinstance(device, torch.device):
        device = select_device(device)
    LOGGER.info(
        f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}"
        f"{'input':>24s}{'output':>24s}"
    )
    gc.collect()  # attempt to free unused memory
    torch.cuda.empty_cache()
    for x in input if isinstance(input, list) else [input]:
        x = x.to(device)
        x.requires_grad = True
        for m in ops if isinstance(ops, list) else [ops]:
            m = m.to(device) if hasattr(m, "to") else m  # device
            m = m.half() if hasattr(m, "half") and isinstance(x, torch.Tensor) and x.dtype is torch.float16 else m
            tf, tb, t = 0, 0, [0, 0, 0]  # dt forward, backward
            try:
                flops = thop.profile(deepcopy(m), inputs=[x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
            except Exception:
                flops = 0

            try:
                mem = 0
                for _ in range(n):
                    with cuda_memory_usage(device) as cuda_info:
                        t[0] = time_sync()
                        y = m(x)
                        t[1] = time_sync()
                        try:
                            (sum(yi.sum() for yi in y) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync()
                        except Exception:  # no backward method
                            # print(e)  # for debug
                            t[2] = float("nan")
                    mem += cuda_info["memory"] / 1e9  # (GB)
                    tf += (t[1] - t[0]) * 1000 / n  # ms per op forward
                    tb += (t[2] - t[1]) * 1000 / n  # ms per op backward
                    if max_num_obj:  # simulate training with predictions per image grid (for AutoBatch)
                        with cuda_memory_usage(device) as cuda_info:
                            torch.randn(
                                x.shape[0],
                                max_num_obj,
                                int(sum((x.shape[-1] / s) * (x.shape[-2] / s) for s in m.stride.tolist())),
                                device=device,
                                dtype=torch.float32,
                            )
                        mem += cuda_info["memory"] / 1e9  # (GB)
                s_in, s_out = (tuple(x.shape) if isinstance(x, torch.Tensor) else "list" for x in (x, y))  # shapes
                p = sum(x.numel() for x in m.parameters()) if isinstance(m, nn.Module) else 0  # parameters
                LOGGER.info(f"{p:12}{flops:12.4g}{mem:>14.3f}{tf:14.4g}{tb:14.4g}{s_in!s:>24s}{s_out!s:>24s}")
                results.append([p, flops, mem, tf, tb, s_in, s_out])
            except Exception as e:
                LOGGER.info(e)
                results.append(None)
            finally:
                gc.collect()  # attempt to free unused memory
                torch.cuda.empty_cache()
    return results
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.torch_utils.attempt_compile` {#ultralytics.utils.torch\_utils.attempt\_compile}

```python
def attempt_compile(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: bool | str = "default",
) -> torch.nn.Module
```

Compile a model with torch.compile and optionally warm up the graph to reduce first-iteration latency.

This utility attempts to compile the provided model using the inductor backend. If compilation is unavailable or fails, the original model is returned unchanged. An optional warmup performs a single forward pass on a dummy input to prime the compiled graph and measure compile/warmup time.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `torch.nn.Module` | Model to compile. | *required* |
| `device` | `torch.device` | Inference device used for warmup and autocast decisions. | *required* |
| `imgsz` | `int, optional` | Square input size to create a dummy tensor with shape (1, 3, imgsz, imgsz) for warmup. | `640` |
| `use_autocast` | `bool, optional` | Whether to run warmup under autocast on CUDA or MPS devices. | `False` |
| `warmup` | `bool, optional` | Whether to execute a single dummy forward pass to warm up the compiled model. | `False` |
| `mode` | `bool | str, optional` | torch.compile mode. True → "default", False → no compile, or a string like<br>    "default", "reduce-overhead", "max-autotune-no-cudagraphs". | `"default"` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.nn.Module` | Compiled model if compilation succeeds, otherwise the original unmodified model. |

**Examples**

```python
>>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
>>> # Try to compile and warm up a model with a 640x640 input
>>> model = attempt_compile(model, device=device, imgsz=640, use_autocast=True, warmup=True)
```

!!! note "Notes"

    - If the current PyTorch build does not provide torch.compile, the function returns the input model immediately.
    - Warmup runs under torch.inference_mode and may use torch.autocast for CUDA/MPS to align compute precision.
    - CUDA devices are synchronized after warmup to account for asynchronous kernel execution.

<details>
<summary>Source code in <code>ultralytics/utils/torch_utils.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py#L911-L987"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def attempt_compile(
    model: torch.nn.Module,
    device: torch.device,
    imgsz: int = 640,
    use_autocast: bool = False,
    warmup: bool = False,
    mode: bool | str = "default",
) -> torch.nn.Module:
    """Compile a model with torch.compile and optionally warm up the graph to reduce first-iteration latency.

    This utility attempts to compile the provided model using the inductor backend. If compilation is unavailable or
    fails, the original model is returned unchanged. An optional warmup performs a single forward pass on a dummy input
    to prime the compiled graph and measure compile/warmup time.

    Args:
        model (torch.nn.Module): Model to compile.
        device (torch.device): Inference device used for warmup and autocast decisions.
        imgsz (int, optional): Square input size to create a dummy tensor with shape (1, 3, imgsz, imgsz) for warmup.
        use_autocast (bool, optional): Whether to run warmup under autocast on CUDA or MPS devices.
        warmup (bool, optional): Whether to execute a single dummy forward pass to warm up the compiled model.
        mode (bool | str, optional): torch.compile mode. True → "default", False → no compile, or a string like
            "default", "reduce-overhead", "max-autotune-no-cudagraphs".

    Returns:
        (torch.nn.Module): Compiled model if compilation succeeds, otherwise the original unmodified model.

    Examples:
        >>> device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        >>> # Try to compile and warm up a model with a 640x640 input
        >>> model = attempt_compile(model, device=device, imgsz=640, use_autocast=True, warmup=True)

    Notes:
        - If the current PyTorch build does not provide torch.compile, the function returns the input model immediately.
        - Warmup runs under torch.inference_mode and may use torch.autocast for CUDA/MPS to align compute precision.
        - CUDA devices are synchronized after warmup to account for asynchronous kernel execution.
    """
    if not hasattr(torch, "compile") or not mode:
        return model

    if mode is True:
        mode = "default"
    prefix = colorstr("compile:")
    LOGGER.info(f"{prefix} starting torch.compile with '{mode}' mode...")
    if mode == "max-autotune":
        LOGGER.warning(f"{prefix} mode='{mode}' not recommended, using mode='max-autotune-no-cudagraphs' instead")
        mode = "max-autotune-no-cudagraphs"
    t0 = time.perf_counter()
    try:
        model = torch.compile(model, mode=mode, backend="inductor")
    except Exception as e:
        LOGGER.warning(f"{prefix} torch.compile failed, continuing uncompiled: {e}")
        return model
    t_compile = time.perf_counter() - t0

    t_warm = 0.0
    if warmup:
        # Use a single dummy tensor to build the graph shape state and reduce first-iteration latency
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        if use_autocast and device.type == "cuda":
            dummy = dummy.half()
        t1 = time.perf_counter()
        with torch.inference_mode():
            if use_autocast and device.type in {"cuda", "mps"}:
                with torch.autocast(device.type):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_warm = time.perf_counter() - t1

    total = t_compile + t_warm
    if warmup:
        LOGGER.info(f"{prefix} complete in {total:.1f}s (compile {t_compile:.1f}s + warmup {t_warm:.1f}s)")
    else:
        LOGGER.info(f"{prefix} compile complete in {t_compile:.1f}s (no warmup)")
    return model
```
</details>

<br><br>
