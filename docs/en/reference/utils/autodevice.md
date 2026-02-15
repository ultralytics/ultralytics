---
description: Discover how to automatically select the most idle GPU using PyTorch and NVIDIA (nvidia-ml-py) with this AutoDevice script. Ideal for optimizing GPU usage in deep learning and computer vision tasks.
keywords: Ultralytics, AutoDevice, GPU selection, PyTorch, NVML, pynvml, GPU monitoring, CUDA, deep learning, idle GPU, memory utilization, temperature, power usage
---

# Reference for `ultralytics/utils/autodevice.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`GPUInfo`](#ultralytics.utils.autodevice.GPUInfo)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`GPUInfo.__del__`](#ultralytics.utils.autodevice.GPUInfo.__del__)
        - [`GPUInfo.shutdown`](#ultralytics.utils.autodevice.GPUInfo.shutdown)
        - [`GPUInfo.refresh_stats`](#ultralytics.utils.autodevice.GPUInfo.refresh_stats)
        - [`GPUInfo._get_device_stats`](#ultralytics.utils.autodevice.GPUInfo._get_device_stats)
        - [`GPUInfo.print_status`](#ultralytics.utils.autodevice.GPUInfo.print_status)
        - [`GPUInfo.select_idle_gpu`](#ultralytics.utils.autodevice.GPUInfo.select_idle_gpu)


## Class `ultralytics.utils.autodevice.GPUInfo` {#ultralytics.utils.autodevice.GPUInfo}

```python
GPUInfo(self)
```

Manages NVIDIA GPU information via pynvml with robust error handling.

Provides methods to query detailed GPU statistics (utilization, memory, temp, power) and select the most idle GPUs based on configurable criteria. It safely handles the absence or initialization failure of the pynvml library by logging warnings and disabling related features, preventing application crashes.

Includes fallback logic using `torch.cuda` for basic device counting if NVML is unavailable during GPU selection. Manages NVML initialization and shutdown internally.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `pynvml` | `module | None` | The `pynvml` module if successfully imported and initialized, otherwise `None`. |
| `nvml_available` | `bool` | Indicates if `pynvml` is ready for use. True if import and `nvmlInit()` succeeded, False<br>    otherwise. |
| `gpu_stats` | `list[dict[str, Any]]` | A list of dictionaries, each holding stats for one GPU, populated on |
| `initialization and by `refresh_stats()`. Keys include: 'index', 'name', 'utilization' (%), 'memory_used' (MiB),<br>    'memory_total' (MiB), 'memory_free' (MiB), 'temperature' (C), 'power_draw' (W), 'power_limit' (W or 'N/A').<br>    Empty if NVML is unavailable or queries fail.` |  |  |

**Methods**

| Name | Description |
| --- | --- |
| [`__del__`](#ultralytics.utils.autodevice.GPUInfo.__del__) | Ensure NVML is shut down when the object is garbage collected. |
| [`_get_device_stats`](#ultralytics.utils.autodevice.GPUInfo._get_device_stats) | Get stats for a single GPU device. |
| [`print_status`](#ultralytics.utils.autodevice.GPUInfo.print_status) | Print GPU status in a compact table format using current stats. |
| [`refresh_stats`](#ultralytics.utils.autodevice.GPUInfo.refresh_stats) | Refresh the internal gpu_stats list by querying NVML. |
| [`select_idle_gpu`](#ultralytics.utils.autodevice.GPUInfo.select_idle_gpu) | Select the most idle GPUs based on utilization and free memory. |
| [`shutdown`](#ultralytics.utils.autodevice.GPUInfo.shutdown) | Shut down NVML if it was initialized. |

**Examples**

```python
Initialize GPUInfo and print status
>>> gpu_info = GPUInfo()
>>> gpu_info.print_status()

Select idle GPUs with minimum memory requirements
>>> selected = gpu_info.select_idle_gpu(count=2, min_memory_fraction=0.2)
>>> print(f"Selected GPU indices: {selected}")
```

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L12-L190"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class GPUInfo:
    """Manages NVIDIA GPU information via pynvml with robust error handling.

    Provides methods to query detailed GPU statistics (utilization, memory, temp, power) and select the most idle GPUs
    based on configurable criteria. It safely handles the absence or initialization failure of the pynvml library by
    logging warnings and disabling related features, preventing application crashes.

    Includes fallback logic using `torch.cuda` for basic device counting if NVML is unavailable during GPU
    selection. Manages NVML initialization and shutdown internally.

    Attributes:
        pynvml (module | None): The `pynvml` module if successfully imported and initialized, otherwise `None`.
        nvml_available (bool): Indicates if `pynvml` is ready for use. True if import and `nvmlInit()` succeeded, False
            otherwise.
        gpu_stats (list[dict[str, Any]]): A list of dictionaries, each holding stats for one GPU, populated on
        initialization and by `refresh_stats()`. Keys include: 'index', 'name', 'utilization' (%), 'memory_used' (MiB),
            'memory_total' (MiB), 'memory_free' (MiB), 'temperature' (C), 'power_draw' (W), 'power_limit' (W or 'N/A').
            Empty if NVML is unavailable or queries fail.

    Methods:
        refresh_stats: Refresh the internal gpu_stats list by querying NVML.
        print_status: Print GPU status in a compact table format using current stats.
        select_idle_gpu: Select the most idle GPUs based on utilization and free memory.
        shutdown: Shut down NVML if it was initialized.

    Examples:
        Initialize GPUInfo and print status
        >>> gpu_info = GPUInfo()
        >>> gpu_info.print_status()

        Select idle GPUs with minimum memory requirements
        >>> selected = gpu_info.select_idle_gpu(count=2, min_memory_fraction=0.2)
        >>> print(f"Selected GPU indices: {selected}")
    """

    def __init__(self):
        """Initialize GPUInfo, attempting to import and initialize pynvml."""
        self.pynvml: Any | None = None
        self.nvml_available: bool = False
        self.gpu_stats: list[dict[str, Any]] = []

        try:
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            self.nvml_available = True
            self.refresh_stats()
        except Exception as e:
            LOGGER.warning(f"Failed to initialize pynvml, GPU stats disabled: {e}")
```
</details>

<br>

### Method `ultralytics.utils.autodevice.GPUInfo.__del__` {#ultralytics.utils.autodevice.GPUInfo.\_\_del\_\_}

```python
def __del__(self)
```

Ensure NVML is shut down when the object is garbage collected.

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L62-L64"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def __del__(self):
    """Ensure NVML is shut down when the object is garbage collected."""
    self.shutdown()
```
</details>

<br>

### Method `ultralytics.utils.autodevice.GPUInfo._get_device_stats` {#ultralytics.utils.autodevice.GPUInfo.\_get\_device\_stats}

```python
def _get_device_stats(self, index: int) -> dict[str, Any]
```

Get stats for a single GPU device.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `index` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L88-L113"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_device_stats(self, index: int) -> dict[str, Any]:
    """Get stats for a single GPU device."""
    handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
    memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)

    def safe_get(func, *args, default=-1, divisor=1):
        try:
            val = func(*args)
            return val // divisor if divisor != 1 and isinstance(val, (int, float)) else val
        except Exception:
            return default

    temp_type = getattr(self.pynvml, "NVML_TEMPERATURE_GPU", -1)

    return {
        "index": index,
        "name": self.pynvml.nvmlDeviceGetName(handle),
        "utilization": util.gpu if util else -1,
        "memory_used": memory.used >> 20 if memory else -1,  # Convert bytes to MiB
        "memory_total": memory.total >> 20 if memory else -1,
        "memory_free": memory.free >> 20 if memory else -1,
        "temperature": safe_get(self.pynvml.nvmlDeviceGetTemperature, handle, temp_type),
        "power_draw": safe_get(self.pynvml.nvmlDeviceGetPowerUsage, handle, divisor=1000),  # Convert mW to W
        "power_limit": safe_get(self.pynvml.nvmlDeviceGetEnforcedPowerLimit, handle, divisor=1000),
    }
```
</details>

<br>

### Method `ultralytics.utils.autodevice.GPUInfo.print_status` {#ultralytics.utils.autodevice.GPUInfo.print\_status}

```python
def print_status(self)
```

Print GPU status in a compact table format using current stats.

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L115-L135"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def print_status(self):
    """Print GPU status in a compact table format using current stats."""
    self.refresh_stats()
    if not self.gpu_stats:
        LOGGER.warning("No GPU stats available.")
        return

    stats = self.gpu_stats
    name_len = max(len(gpu.get("name", "N/A")) for gpu in stats)
    hdr = f"{'Idx':<3} {'Name':<{name_len}} {'Util':>6} {'Mem (MiB)':>15} {'Temp':>5} {'Pwr (W)':>10}"
    LOGGER.info(f"\n--- GPU Status ---\n{hdr}\n{'-' * len(hdr)}")

    for gpu in stats:
        u = f"{gpu['utilization']:>5}%" if gpu["utilization"] >= 0 else " N/A "
        m = f"{gpu['memory_used']:>6}/{gpu['memory_total']:<6}" if gpu["memory_used"] >= 0 else " N/A / N/A "
        t = f"{gpu['temperature']}C" if gpu["temperature"] >= 0 else " N/A "
        p = f"{gpu['power_draw']:>3}/{gpu['power_limit']:<3}" if gpu["power_draw"] >= 0 else " N/A "

        LOGGER.info(f"{gpu.get('index'):<3d} {gpu.get('name', 'N/A'):<{name_len}} {u:>6} {m:>15} {t:>5} {p:>10}")

    LOGGER.info(f"{'-' * len(hdr)}\n")
```
</details>

<br>

### Method `ultralytics.utils.autodevice.GPUInfo.refresh_stats` {#ultralytics.utils.autodevice.GPUInfo.refresh\_stats}

```python
def refresh_stats(self)
```

Refresh the internal gpu_stats list by querying NVML.

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L75-L86"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def refresh_stats(self):
    """Refresh the internal gpu_stats list by querying NVML."""
    self.gpu_stats = []
    if not self.nvml_available or not self.pynvml:
        return

    try:
        device_count = self.pynvml.nvmlDeviceGetCount()
        self.gpu_stats.extend(self._get_device_stats(i) for i in range(device_count))
    except Exception as e:
        LOGGER.warning(f"Error during device query: {e}")
        self.gpu_stats = []
```
</details>

<br>

### Method `ultralytics.utils.autodevice.GPUInfo.select_idle_gpu` {#ultralytics.utils.autodevice.GPUInfo.select\_idle\_gpu}

```python
def select_idle_gpu(self, count: int = 1, min_memory_fraction: float = 0, min_util_fraction: float = 0) -> list[int]
```

Select the most idle GPUs based on utilization and free memory.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `count` | `int` | The number of idle GPUs to select. | `1` |
| `min_memory_fraction` | `float` | Minimum free memory required as a fraction of total memory. | `0` |
| `min_util_fraction` | `float` | Minimum free utilization rate required from 0.0 - 1.0. | `0` |

**Returns**

| Type | Description |
| --- | --- |
| `list[int]` | Indices of the selected GPUs, sorted by idleness (lowest utilization first). |

!!! note "Notes"

    Returns fewer than 'count' if not enough qualify or exist.
    Returns empty list if NVML stats are unavailable or no GPUs meet the criteria.

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L137-L190"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def select_idle_gpu(
    self, count: int = 1, min_memory_fraction: float = 0, min_util_fraction: float = 0
) -> list[int]:
    """Select the most idle GPUs based on utilization and free memory.

    Args:
        count (int): The number of idle GPUs to select.
        min_memory_fraction (float): Minimum free memory required as a fraction of total memory.
        min_util_fraction (float): Minimum free utilization rate required from 0.0 - 1.0.

    Returns:
        (list[int]): Indices of the selected GPUs, sorted by idleness (lowest utilization first).

    Notes:
         Returns fewer than 'count' if not enough qualify or exist.
         Returns empty list if NVML stats are unavailable or no GPUs meet the criteria.
    """
    assert min_memory_fraction <= 1.0, f"min_memory_fraction must be <= 1.0, got {min_memory_fraction}"
    assert min_util_fraction <= 1.0, f"min_util_fraction must be <= 1.0, got {min_util_fraction}"
    criteria = (
        f"free memory >= {min_memory_fraction * 100:.1f}% and free utilization >= {min_util_fraction * 100:.1f}%"
    )
    LOGGER.info(f"Searching for {count} idle GPUs with {criteria}...")

    if count <= 0:
        return []

    self.refresh_stats()
    if not self.gpu_stats:
        LOGGER.warning("NVML stats unavailable.")
        return []

    # Filter and sort eligible GPUs
    eligible_gpus = [
        gpu
        for gpu in self.gpu_stats
        if gpu.get("memory_free", 0) / gpu.get("memory_total", 1) >= min_memory_fraction
        and (100 - gpu.get("utilization", 100)) >= min_util_fraction * 100
    ]
    # Random tiebreaker prevents race conditions when multiple processes start simultaneously
    # and all GPUs appear equally idle (same utilization and free memory)
    eligible_gpus.sort(key=lambda x: (x.get("utilization", 101), -x.get("memory_free", 0), random.random()))

    # Select top 'count' indices
    selected = [gpu["index"] for gpu in eligible_gpus[:count]]

    if selected:
        if len(selected) < count:
            LOGGER.warning(f"Requested {count} GPUs but only {len(selected)} met the idle criteria.")
        LOGGER.info(f"Selected idle CUDA devices {selected}")
    else:
        LOGGER.warning(f"No GPUs met criteria ({criteria}).")

    return selected
```
</details>

<br>

### Method `ultralytics.utils.autodevice.GPUInfo.shutdown` {#ultralytics.utils.autodevice.GPUInfo.shutdown}

```python
def shutdown(self)
```

Shut down NVML if it was initialized.

<details>
<summary>Source code in <code>ultralytics/utils/autodevice.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/autodevice.py#L66-L73"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def shutdown(self):
    """Shut down NVML if it was initialized."""
    if self.nvml_available and self.pynvml:
        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass
        self.nvml_available = False
```
</details>

<br><br>
