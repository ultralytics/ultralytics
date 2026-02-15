---
description: High-performance console output capture with API/file streaming for YOLO11 training logs.
keywords: ConsoleLogger, console capture, log streaming, API logging, file logging, YOLO11, Ultralytics
---

# Reference for `ultralytics/utils/logger.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ConsoleLogger`](#ultralytics.utils.logger.ConsoleLogger)
        - [`SystemLogger`](#ultralytics.utils.logger.SystemLogger)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ConsoleLogger.start_capture`](#ultralytics.utils.logger.ConsoleLogger.start_capture)
        - [`ConsoleLogger.stop_capture`](#ultralytics.utils.logger.ConsoleLogger.stop_capture)
        - [`ConsoleLogger._queue_log`](#ultralytics.utils.logger.ConsoleLogger._queue_log)
        - [`ConsoleLogger._flush_worker`](#ultralytics.utils.logger.ConsoleLogger._flush_worker)
        - [`ConsoleLogger._flush_buffer`](#ultralytics.utils.logger.ConsoleLogger._flush_buffer)
        - [`ConsoleLogger._write_destination`](#ultralytics.utils.logger.ConsoleLogger._write_destination)
        - [`SystemLogger._init_nvidia`](#ultralytics.utils.logger.SystemLogger._init_nvidia)
        - [`SystemLogger.get_metrics`](#ultralytics.utils.logger.SystemLogger.get_metrics)
        - [`SystemLogger._get_nvidia_metrics`](#ultralytics.utils.logger.SystemLogger._get_nvidia_metrics)


## Class `ultralytics.utils.logger.ConsoleLogger` {#ultralytics.utils.logger.ConsoleLogger}

```python
ConsoleLogger(self, destination = None, batch_size = 1, flush_interval = 5.0, on_flush = None)
```

Console output capture with batched streaming to file, API, or custom callback.

Captures stdout/stderr output and streams it with intelligent deduplication and configurable batching.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `destination` | `str | Path | None` | API endpoint URL (http/https), local file path, or None. | `None` |
| `batch_size` | `int` | Lines to accumulate before flush (1 = immediate, higher = batched). | `1` |
| `flush_interval` | `float` | Max seconds between flushes when batching. | `5.0` |
| `on_flush` | `callable | None` | Callback(content: str, line_count: int, chunk_id: int) for custom handling. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `destination` | `str | Path | None` | Target destination for streaming (URL, Path, or None for callback-only). |
| `batch_size` | `int` | Number of lines to batch before flushing (default: 1 for immediate). |
| `flush_interval` | `float` | Seconds between automatic flushes (default: 5.0). |
| `on_flush` | `callable | None` | Optional callback function called with batched content on flush. |
| `active` | `bool` | Whether console capture is currently active. |

**Methods**

| Name | Description |
| --- | --- |
| [`_flush_buffer`](#ultralytics.utils.logger.ConsoleLogger._flush_buffer) | Flush buffered lines to destination and/or callback. |
| [`_flush_worker`](#ultralytics.utils.logger.ConsoleLogger._flush_worker) | Background worker that flushes buffer periodically. |
| [`_queue_log`](#ultralytics.utils.logger.ConsoleLogger._queue_log) | Queue console text with deduplication and timestamp processing. |
| [`_write_destination`](#ultralytics.utils.logger.ConsoleLogger._write_destination) | Write content to file or API destination. |
| [`start_capture`](#ultralytics.utils.logger.ConsoleLogger.start_capture) | Start capturing console output and redirect stdout/stderr. |
| [`stop_capture`](#ultralytics.utils.logger.ConsoleLogger.stop_capture) | Stop capturing console output and flush remaining buffer. |

**Examples**

```python
File logging (immediate):
>>> logger = ConsoleLogger("training.log")
>>> logger.start_capture()
>>> print("This will be logged")
>>> logger.stop_capture()

API streaming with batching:
>>> logger = ConsoleLogger("https://api.example.com/logs", batch_size=10)
>>> logger.start_capture()

Custom callback with batching:
>>> def my_handler(content, line_count, chunk_id):
...     print(f"Received {line_count} lines")
>>> logger = ConsoleLogger(on_flush=my_handler, batch_size=5)
>>> logger.start_capture()
```

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L15-L285"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class ConsoleLogger:
    """Console output capture with batched streaming to file, API, or custom callback.

    Captures stdout/stderr output and streams it with intelligent deduplication and configurable batching.

    Attributes:
        destination (str | Path | None): Target destination for streaming (URL, Path, or None for callback-only).
        batch_size (int): Number of lines to batch before flushing (default: 1 for immediate).
        flush_interval (float): Seconds between automatic flushes (default: 5.0).
        on_flush (callable | None): Optional callback function called with batched content on flush.
        active (bool): Whether console capture is currently active.

    Examples:
        File logging (immediate):
        >>> logger = ConsoleLogger("training.log")
        >>> logger.start_capture()
        >>> print("This will be logged")
        >>> logger.stop_capture()

        API streaming with batching:
        >>> logger = ConsoleLogger("https://api.example.com/logs", batch_size=10)
        >>> logger.start_capture()

        Custom callback with batching:
        >>> def my_handler(content, line_count, chunk_id):
        ...     print(f"Received {line_count} lines")
        >>> logger = ConsoleLogger(on_flush=my_handler, batch_size=5)
        >>> logger.start_capture()
    """

    def __init__(self, destination=None, batch_size=1, flush_interval=5.0, on_flush=None):
        """Initialize console logger with optional batching.

        Args:
            destination (str | Path | None): API endpoint URL (http/https), local file path, or None.
            batch_size (int): Lines to accumulate before flush (1 = immediate, higher = batched).
            flush_interval (float): Max seconds between flushes when batching.
            on_flush (callable | None): Callback(content: str, line_count: int, chunk_id: int) for custom handling.
        """
        self.destination = destination
        self.is_api = isinstance(destination, str) and destination.startswith(("http://", "https://"))
        if destination is not None and not self.is_api:
            self.destination = Path(destination)

        # Batching configuration
        self.batch_size = max(1, batch_size)
        self.flush_interval = flush_interval
        self.on_flush = on_flush

        # Console capture state
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.active = False
        self._log_handler = None  # Track handler for cleanup

        # Buffer for batching
        self.buffer = []
        self.buffer_lock = threading.Lock()
        self.flush_thread = None
        self.chunk_id = 0

        # Deduplication state
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_line = ""  # Track progress sequence key for deduplication
        self.last_was_progress = False  # Track if last line was a progress bar
```
</details>

<br>

### Method `ultralytics.utils.logger.ConsoleLogger._flush_buffer` {#ultralytics.utils.logger.ConsoleLogger.\_flush\_buffer}

```python
def _flush_buffer(self)
```

Flush buffered lines to destination and/or callback.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L215-L237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _flush_buffer(self):
    """Flush buffered lines to destination and/or callback."""
    with self.buffer_lock:
        if not self.buffer:
            return
        lines = self.buffer.copy()
        self.buffer.clear()
        self.chunk_id += 1
        chunk_id = self.chunk_id  # Capture under lock to avoid race

    content = "\n".join(lines)
    line_count = len(lines)

    # Call custom callback if provided
    if self.on_flush:
        try:
            self.on_flush(content, line_count, chunk_id)
        except Exception:
            pass  # Silently ignore callback errors to avoid flooding stderr

    # Write to destination (file or API)
    if self.destination is not None:
        self._write_destination(content)
```
</details>

<br>

### Method `ultralytics.utils.logger.ConsoleLogger._flush_worker` {#ultralytics.utils.logger.ConsoleLogger.\_flush\_worker}

```python
def _flush_worker(self)
```

Background worker that flushes buffer periodically.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L208-L213"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _flush_worker(self):
    """Background worker that flushes buffer periodically."""
    while self.active:
        time.sleep(self.flush_interval)
        if self.active:
            self._flush_buffer()
```
</details>

<br>

### Method `ultralytics.utils.logger.ConsoleLogger._queue_log` {#ultralytics.utils.logger.ConsoleLogger.\_queue\_log}

```python
def _queue_log(self, text)
```

Queue console text with deduplication and timestamp processing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `text` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L127-L206"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _queue_log(self, text):
    """Queue console text with deduplication and timestamp processing."""
    if not self.active:
        return

    current_time = time.time()

    # Handle carriage returns and process lines
    if "\r" in text:
        text = text.split("\r")[-1]

    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    for line in lines:
        line = line.rstrip()

        # Skip lines with only thin progress bars (partial progress)
        if "─" in line:  # Has thin lines but no thick lines
            continue

        # Only show 100% completion lines for progress bars
        if " ━━" in line:
            is_complete = "100%" in line

            # Skip ALL non-complete progress lines
            if not is_complete:
                continue

            # Extract sequence key to deduplicate multiple 100% lines for same sequence
            parts = line.split()
            seq_key = ""
            if parts:
                # Check for epoch pattern (X/Y at start)
                if "/" in parts[0] and parts[0].replace("/", "").isdigit():
                    seq_key = parts[0]  # e.g., "1/3"
                elif parts[0] == "Class" and len(parts) > 1:
                    seq_key = f"{parts[0]}_{parts[1]}"  # e.g., "Class_train:" or "Class_val:"
                elif parts[0] in ("train:", "val:"):
                    seq_key = parts[0]  # Phase identifier

            # Skip if we already showed 100% for this sequence
            if seq_key and self.last_progress_line == f"{seq_key}:done":
                continue

            # Mark this sequence as done
            if seq_key:
                self.last_progress_line = f"{seq_key}:done"

            self.last_was_progress = True
        else:
            # Skip empty line after progress bar
            if not line and self.last_was_progress:
                self.last_was_progress = False
                continue
            self.last_was_progress = False

        # General deduplication
        if line == self.last_line and current_time - self.last_time < 0.1:
            continue

        self.last_line = line
        self.last_time = current_time

        # Add timestamp if needed
        if not line.startswith("[20"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {line}"

        # Add to buffer and check if flush needed
        should_flush = False
        with self.buffer_lock:
            self.buffer.append(line)
            if len(self.buffer) >= self.batch_size:
                should_flush = True

        # Flush outside lock to avoid deadlock
        if should_flush:
            self._flush_buffer()
```
</details>

<br>

### Method `ultralytics.utils.logger.ConsoleLogger._write_destination` {#ultralytics.utils.logger.ConsoleLogger.\_write\_destination}

```python
def _write_destination(self, content)
```

Write content to file or API destination.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `content` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L239-L252"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _write_destination(self, content):
    """Write content to file or API destination."""
    try:
        if self.is_api:
            import requests

            payload = {"timestamp": datetime.now().isoformat(), "message": content}
            requests.post(str(self.destination), json=payload, timeout=5)
        else:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            with self.destination.open("a", encoding="utf-8") as f:
                f.write(content + "\n")
    except Exception as e:
        print(f"Console logger write error: {e}", file=self.original_stderr)
```
</details>

<br>

### Method `ultralytics.utils.logger.ConsoleLogger.start_capture` {#ultralytics.utils.logger.ConsoleLogger.start\_capture}

```python
def start_capture(self)
```

Start capturing console output and redirect stdout/stderr.

!!! note "Notes"

    In DDP training, only activates on rank 0/-1 to prevent duplicate logging.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L82-L105"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def start_capture(self):
    """Start capturing console output and redirect stdout/stderr.

    Notes:
        In DDP training, only activates on rank 0/-1 to prevent duplicate logging.
    """
    if self.active or RANK not in {-1, 0}:
        return

    self.active = True
    sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
    sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)

    # Hook Ultralytics logger
    try:
        self._log_handler = self._LogHandler(self._queue_log)
        logging.getLogger("ultralytics").addHandler(self._log_handler)
    except Exception:
        pass

    # Start background flush thread for batched mode
    if self.batch_size > 1:
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
```
</details>

<br>

### Method `ultralytics.utils.logger.ConsoleLogger.stop_capture` {#ultralytics.utils.logger.ConsoleLogger.stop\_capture}

```python
def stop_capture(self)
```

Stop capturing console output and flush remaining buffer.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L107-L125"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def stop_capture(self):
    """Stop capturing console output and flush remaining buffer."""
    if not self.active:
        return

    self.active = False
    sys.stdout = self.original_stdout
    sys.stderr = self.original_stderr

    # Remove logging handler to prevent memory leak
    if self._log_handler:
        try:
            logging.getLogger("ultralytics").removeHandler(self._log_handler)
        except Exception:
            pass
        self._log_handler = None

    # Final flush
    self._flush_buffer()
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.logger.SystemLogger` {#ultralytics.utils.logger.SystemLogger}

```python
SystemLogger(self)
```

Log dynamic system metrics for training monitoring.

Captures real-time system metrics including CPU, RAM, disk I/O, network I/O, and NVIDIA GPU statistics for training performance monitoring and analysis.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `pynvml` |  | NVIDIA pynvml module instance if successfully imported, None otherwise. |
| `nvidia_initialized` | `bool` | Whether NVIDIA GPU monitoring is available and initialized. |
| `net_start` |  | Initial network I/O counters for calculating cumulative usage. |
| `disk_start` |  | Initial disk I/O counters for calculating cumulative usage. |

**Methods**

| Name | Description |
| --- | --- |
| [`_get_nvidia_metrics`](#ultralytics.utils.logger.SystemLogger._get_nvidia_metrics) | Get NVIDIA GPU metrics including utilization, memory, temperature, and power. |
| [`_init_nvidia`](#ultralytics.utils.logger.SystemLogger._init_nvidia) | Initialize NVIDIA GPU monitoring with pynvml. |
| [`get_metrics`](#ultralytics.utils.logger.SystemLogger.get_metrics) | Get current system metrics including CPU, RAM, disk, network, and GPU usage. |

**Examples**

```python
Basic usage:
>>> logger = SystemLogger()
>>> metrics = logger.get_metrics()
>>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
>>> if metrics["gpus"]:
...     gpu0 = metrics["gpus"]["0"]
...     print(f"GPU0: {gpu0['usage']}% usage, {gpu0['temp']}°C")

Training loop integration:
>>> system_logger = SystemLogger()
>>> for epoch in range(epochs):
...     # Training code here
...     metrics = system_logger.get_metrics()
...     # Log to database/file
```

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L288-L465"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class SystemLogger:
    """Log dynamic system metrics for training monitoring.

    Captures real-time system metrics including CPU, RAM, disk I/O, network I/O, and NVIDIA GPU statistics for training
    performance monitoring and analysis.

    Attributes:
        pynvml: NVIDIA pynvml module instance if successfully imported, None otherwise.
        nvidia_initialized (bool): Whether NVIDIA GPU monitoring is available and initialized.
        net_start: Initial network I/O counters for calculating cumulative usage.
        disk_start: Initial disk I/O counters for calculating cumulative usage.

    Examples:
        Basic usage:
        >>> logger = SystemLogger()
        >>> metrics = logger.get_metrics()
        >>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
        >>> if metrics["gpus"]:
        ...     gpu0 = metrics["gpus"]["0"]
        ...     print(f"GPU0: {gpu0['usage']}% usage, {gpu0['temp']}°C")

        Training loop integration:
        >>> system_logger = SystemLogger()
        >>> for epoch in range(epochs):
        ...     # Training code here
        ...     metrics = system_logger.get_metrics()
        ...     # Log to database/file
    """

    def __init__(self):
        """Initialize the system logger."""
        import psutil  # scoped as slow import

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()

        # For rate calculation
        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()
```
</details>

<br>

### Method `ultralytics.utils.logger.SystemLogger._get_nvidia_metrics` {#ultralytics.utils.logger.SystemLogger.\_get\_nvidia\_metrics}

```python
def _get_nvidia_metrics(self)
```

Get NVIDIA GPU metrics including utilization, memory, temperature, and power.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L443-L465"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_nvidia_metrics(self):
    """Get NVIDIA GPU metrics including utilization, memory, temperature, and power."""
    gpus = {}
    if not self.nvidia_initialized or not self.pynvml:
        return gpus
    try:
        device_count = self.pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
            util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
            power = self.pynvml.nvmlDeviceGetPowerUsage(handle) // 1000

            gpus[str(i)] = {
                "usage": round(util.gpu, 3),
                "memory": round((memory.used / memory.total) * 100, 3),
                "temp": temp,
                "power": power,
            }
    except Exception:
        pass
    return gpus
```
</details>

<br>

### Method `ultralytics.utils.logger.SystemLogger._init_nvidia` {#ultralytics.utils.logger.SystemLogger.\_init\_nvidia}

```python
def _init_nvidia(self)
```

Initialize NVIDIA GPU monitoring with pynvml.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L331-L346"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _init_nvidia(self):
    """Initialize NVIDIA GPU monitoring with pynvml."""
    if MACOS:
        return False

    try:
        check_requirements("nvidia-ml-py>=12.0.0")
        self.pynvml = __import__("pynvml")
        self.pynvml.nvmlInit()
        return True
    except Exception as e:
        import torch

        if torch.cuda.is_available():
            LOGGER.warning(f"SystemLogger NVML init failed: {e}")
        return False
```
</details>

<br>

### Method `ultralytics.utils.logger.SystemLogger.get_metrics` {#ultralytics.utils.logger.SystemLogger.get\_metrics}

```python
def get_metrics(self, rates = False)
```

Get current system metrics including CPU, RAM, disk, network, and GPU usage.

Collects comprehensive system metrics including CPU usage, RAM usage, disk I/O statistics, network I/O
statistics, and GPU metrics (if available).

Example output (rates=False, default):
```python
{
    "cpu": 45.2,
    "ram": 78.9,
    "disk": {"read_mb": 156.7, "write_mb": 89.3, "used_gb": 256.8},
    "network": {"recv_mb": 157.2, "sent_mb": 89.1},
    "gpus": {
        "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
        "1": {"usage": 94.1, "memory": 82.7, "temp": 70, "power": 278},
    },
}
```

Example output (rates=True):
```python
{
    "cpu": 45.2,
    "ram": 78.9,
    "disk": {"read_mbs": 12.5, "write_mbs": 8.3, "used_gb": 256.8},
    "network": {"recv_mbs": 5.2, "sent_mbs": 1.1},
    "gpus": {
        "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
    },
}
```

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `rates` | `bool` | If True, return disk/network as MB/s rates instead of cumulative MB. | `False` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Metrics dictionary with cpu, ram, disk, network, and gpus keys. |

**Examples**

```python
>>> logger = SystemLogger()
>>> logger.get_metrics()["cpu"]  # CPU percentage
>>> logger.get_metrics(rates=True)["network"]["recv_mbs"]  # MB/s download rate
```

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L348-L441"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_metrics(self, rates=False):
    """Get current system metrics including CPU, RAM, disk, network, and GPU usage.

    Collects comprehensive system metrics including CPU usage, RAM usage, disk I/O statistics, network I/O
    statistics, and GPU metrics (if available).

    Example output (rates=False, default):
    ```python
    {
        "cpu": 45.2,
        "ram": 78.9,
        "disk": {"read_mb": 156.7, "write_mb": 89.3, "used_gb": 256.8},
        "network": {"recv_mb": 157.2, "sent_mb": 89.1},
        "gpus": {
            "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
            "1": {"usage": 94.1, "memory": 82.7, "temp": 70, "power": 278},
        },
    }
    ```

    Example output (rates=True):
    ```python
    {
        "cpu": 45.2,
        "ram": 78.9,
        "disk": {"read_mbs": 12.5, "write_mbs": 8.3, "used_gb": 256.8},
        "network": {"recv_mbs": 5.2, "sent_mbs": 1.1},
        "gpus": {
            "0": {"usage": 95.6, "memory": 85.4, "temp": 72, "power": 285},
        },
    }
    ```

    Args:
        rates (bool): If True, return disk/network as MB/s rates instead of cumulative MB.

    Returns:
        (dict): Metrics dictionary with cpu, ram, disk, network, and gpus keys.

    Examples:
        >>> logger = SystemLogger()
        >>> logger.get_metrics()["cpu"]  # CPU percentage
        >>> logger.get_metrics(rates=True)["network"]["recv_mbs"]  # MB/s download rate
    """
    import psutil  # scoped as slow import

    net = psutil.net_io_counters()
    disk = psutil.disk_io_counters()
    memory = psutil.virtual_memory()
    disk_usage = shutil.disk_usage("/")
    now = time.time()

    metrics = {
        "cpu": round(psutil.cpu_percent(), 3),
        "ram": round(memory.percent, 3),
        "gpus": {},
    }

    # Calculate elapsed time since last call
    elapsed = max(0.1, now - self._prev_time)  # Avoid division by zero

    if rates:
        # Calculate MB/s rates from delta since last call
        metrics["disk"] = {
            "read_mbs": round(max(0, (disk.read_bytes - self._prev_disk.read_bytes) / (1 << 20) / elapsed), 3),
            "write_mbs": round(max(0, (disk.write_bytes - self._prev_disk.write_bytes) / (1 << 20) / elapsed), 3),
            "used_gb": round(disk_usage.used / (1 << 30), 3),
        }
        metrics["network"] = {
            "recv_mbs": round(max(0, (net.bytes_recv - self._prev_net.bytes_recv) / (1 << 20) / elapsed), 3),
            "sent_mbs": round(max(0, (net.bytes_sent - self._prev_net.bytes_sent) / (1 << 20) / elapsed), 3),
        }
    else:
        # Cumulative MB since initialization (original behavior)
        metrics["disk"] = {
            "read_mb": round((disk.read_bytes - self.disk_start.read_bytes) / (1 << 20), 3),
            "write_mb": round((disk.write_bytes - self.disk_start.write_bytes) / (1 << 20), 3),
            "used_gb": round(disk_usage.used / (1 << 30), 3),
        }
        metrics["network"] = {
            "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / (1 << 20), 3),
            "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / (1 << 20), 3),
        }

    # Always update previous values for accurate rate calculation on next call
    self._prev_net = net
    self._prev_disk = disk
    self._prev_time = now

    # Add GPU metrics (NVIDIA only)
    if self.nvidia_initialized:
        metrics["gpus"].update(self._get_nvidia_metrics())

    return metrics
```
</details>

<br><br>
