---
title: utils.logger API Reference
description: High-performance console output capture with API/file streaming for YOLO11 training logs.
keywords: ConsoleLogger, console capture, log streaming, API logging, file logging, YOLO11, Ultralytics
---

# Reference for `ultralytics/utils/logger.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`ConsoleLogger`](#ultralytics.utils.logger.ConsoleLogger)
        - [`_DriveInfo`](#ultralytics.utils.logger._DriveInfo)
        - [`SystemLogger`](#ultralytics.utils.logger.SystemLogger)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`ConsoleLogger.start_capture`](#ultralytics.utils.logger.ConsoleLogger.start_capture)
        - [`ConsoleLogger.stop_capture`](#ultralytics.utils.logger.ConsoleLogger.stop_capture)
        - [`ConsoleLogger._queue_log`](#ultralytics.utils.logger.ConsoleLogger._queue_log)
        - [`ConsoleLogger._flush_worker`](#ultralytics.utils.logger.ConsoleLogger._flush_worker)
        - [`ConsoleLogger._flush_buffer`](#ultralytics.utils.logger.ConsoleLogger._flush_buffer)
        - [`ConsoleLogger._write_destination`](#ultralytics.utils.logger.ConsoleLogger._write_destination)
        - [`_DriveInfo.mounts`](#ultralytics.utils.logger._DriveInfo.mounts)
        - [`_DriveInfo._sort`](#ultralytics.utils.logger._DriveInfo._sort)
        - [`_DriveInfo._current_mount`](#ultralytics.utils.logger._DriveInfo._current_mount)
        - [`_DriveInfo._macos_mounts`](#ultralytics.utils.logger._DriveInfo._macos_mounts)
        - [`_DriveInfo._linux_mounts`](#ultralytics.utils.logger._DriveInfo._linux_mounts)
        - [`_DriveInfo._windows_mounts`](#ultralytics.utils.logger._DriveInfo._windows_mounts)
        - [`SystemLogger._init_nvidia`](#ultralytics.utils.logger.SystemLogger._init_nvidia)
        - [`SystemLogger.get_metrics`](#ultralytics.utils.logger.SystemLogger.get_metrics)
        - [`SystemLogger._get_nvidia_metrics`](#ultralytics.utils.logger.SystemLogger._get_nvidia_metrics)


## Class `ultralytics.utils.logger.ConsoleLogger` {#ultralytics.utils.logger.ConsoleLogger}

```python
ConsoleLogger(destination=None, batch_size=1, flush_interval=5.0, on_flush=None)
```

Console output capture with batched streaming to file, API, or custom callback.

Captures stdout/stderr output and streams it with intelligent deduplication and configurable batching.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `destination` | `str \| Path \| None` | API endpoint URL (http/https), local file path, or None. | `None` |
| `batch_size` | `int` | Lines to accumulate before flush (1 = immediate, higher = batched). | `1` |
| `flush_interval` | `float` | Max seconds between flushes when batching. | `5.0` |
| `on_flush` | `callable \| None` | Callback(content: str, line_count: int, chunk_id: int) for custom handling. | `None` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `destination` | `str \| Path \| None` | Target destination for streaming (URL, Path, or None for callback-only). |
| `batch_size` | `int` | Number of lines to batch before flushing (default: 1 for immediate). |
| `flush_interval` | `float` | Seconds between automatic flushes (default: 5.0). |
| `on_flush` | `callable \| None` | Optional callback function called with batched content on flush. |
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

File logging (immediate):

```python
>>> logger = ConsoleLogger("training.log")
>>> logger.start_capture()
>>> print("This will be logged")
>>> logger.stop_capture()
```

API streaming with batching:

```python
>>> logger = ConsoleLogger("https://api.example.com/logs", batch_size=10)
>>> logger.start_capture()
```

Custom callback with batching:

```python
>>> def my_handler(content, line_count, chunk_id):
...     print(f"Received {line_count} lines")
>>> logger = ConsoleLogger(on_flush=my_handler, batch_size=5)
>>> logger.start_capture()
```

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L17-L294">View on GitHub</a>
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
        if isinstance(destination, str) and destination.startswith("http://"):
            LOGGER.warning("ConsoleLogger destination uses plaintext HTTP; captured logs are sent unencrypted.")
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L220-L242">View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L213-L218">View on GitHub</a>
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

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L131-L211">View on GitHub</a>
```python
def _queue_log(self, text):
    """Queue console text with deduplication and timestamp processing."""
    if not self.active:
        return

    current_time = time.time()

    # Handle carriage returns and strip ANSI clear-line codes (TQDM writes "\r\033[K<line>" interactively)
    if "\r" in text:
        text = text.split("\r")[-1]
    text = text.replace("\x1b[K", "")

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
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
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

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L244-L257">View on GitHub</a>
```python
def _write_destination(self, content):
    """Write content to file or API destination."""
    try:
        if self.is_api:
            import requests

            payload = {"timestamp": datetime.now().astimezone().isoformat(), "message": content}
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L86-L109">View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L111-L129">View on GitHub</a>
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

## Class `ultralytics.utils.logger._DriveInfo` {#ultralytics.utils.logger.\_DriveInfo}

```python
_DriveInfo()
```

Resolve mounted storage paths backed by local drives.

This helper keeps platform-specific drive discovery isolated from SystemLogger metric collection. It uses fast psutil mount discovery first and falls back to native OS commands only when multiple visible mounts need disambiguation.

**Methods**

| Name | Description |
| --- | --- |
| [`_current_mount`](#ultralytics.utils.logger._DriveInfo._current_mount) | Get the mounted filesystem backing the current working directory. |
| [`_linux_mounts`](#ultralytics.utils.logger._DriveInfo._linux_mounts) | Get Linux mounts backed by physical block devices. |
| [`_macos_mounts`](#ultralytics.utils.logger._DriveInfo._macos_mounts) | Get user-visible macOS mounts backed by physical disks. |
| [`_sort`](#ultralytics.utils.logger._DriveInfo._sort) | Sort mounted paths with root first, excluding boot/firmware partitions like /boot and /boot/efi. |
| [`_windows_mounts`](#ultralytics.utils.logger._DriveInfo._windows_mounts) | Get Windows fixed local drive mounts. |
| [`mounts`](#ultralytics.utils.logger._DriveInfo.mounts) | Get mounted paths to monitor. |

**Examples**

```python
>>> logger = SystemLogger(all_drives=True)
>>> logger.mounts
['/']
```

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L297-L427">View on GitHub</a>
```python
class _DriveInfo:
    """Resolve mounted storage paths backed by local drives.

    This helper keeps platform-specific drive discovery isolated from SystemLogger metric collection. It uses fast
    psutil mount discovery first and falls back to native OS commands only when multiple visible mounts need
    disambiguation.

    Examples:
        >>> logger = SystemLogger(all_drives=True)
        >>> logger.mounts
        ['/']
    """
```
</details>

<br>

### Method `ultralytics.utils.logger._DriveInfo._current_mount` {#ultralytics.utils.logger.\_DriveInfo.\_current\_mount}

```python
def _current_mount(partitions)
```

Get the mounted filesystem backing the current working directory.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L343-L357">View on GitHub</a>
```python
@staticmethod
def _current_mount(partitions):
    """Get the mounted filesystem backing the current working directory."""
    try:
        cwd = Path.cwd().resolve()
    except OSError:
        return "C:\\" if WINDOWS else "/"
    matches = []
    for partition in partitions:
        try:
            mount = Path(partition.mountpoint).resolve()
        except OSError:
            continue
        if cwd == mount or cwd.is_relative_to(mount):
            matches.append(partition.mountpoint)
    return max(matches, key=len, default=Path.cwd().anchor or "/")
```
</details>

<br>

### Method `ultralytics.utils.logger._DriveInfo._linux_mounts` {#ultralytics.utils.logger.\_DriveInfo.\_linux\_mounts}

```python
def _linux_mounts(_partitions)
```

Get Linux mounts backed by physical block devices.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L391-L412">View on GitHub</a>
```python
@staticmethod
def _linux_mounts(_partitions):
    """Get Linux mounts backed by physical block devices."""
    block_info = json.loads(
        subprocess.check_output(
            ["lsblk", "--json", "--output", "NAME,TYPE,MOUNTPOINT,MOUNTPOINTS"], text=True, timeout=5
        )
    )
    mounts = []

    def visit(block, physical=False):
        physical = physical or block.get("type") == "disk"
        if physical:
            values = block.get("mountpoints") or [block.get("mountpoint")]
            if isinstance(values, str):
                values = [values]
            mounts.extend(m for m in values if isinstance(m, str) and m.startswith("/") and Path(m).is_dir())
        for child in block.get("children", []):
            visit(child, physical)

    for block in block_info.get("blockdevices", []):
        visit(block)
    return mounts
```
</details>

<br>

### Method `ultralytics.utils.logger._DriveInfo._macos_mounts` {#ultralytics.utils.logger.\_DriveInfo.\_macos\_mounts}

```python
def _macos_mounts(partitions)
```

Get user-visible macOS mounts backed by physical disks.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L360-L388">View on GitHub</a>
```python
@staticmethod
def _macos_mounts(partitions):
    """Get user-visible macOS mounts backed by physical disks."""
    disk_info = plistlib.loads(subprocess.check_output(["diskutil", "list", "-plist", "physical"], timeout=5))
    physical_devices = set(disk_info.get("WholeDisks", []))
    for disk in disk_info.get("AllDisksAndPartitions", []):
        physical_devices.add(disk.get("DeviceIdentifier", ""))
        physical_devices.update(p.get("DeviceIdentifier", "") for p in disk.get("Partitions", []))

    mounts, volume_groups = [], set()
    for partition in partitions:
        if partition.mountpoint != "/" and "dontbrowse" in partition.opts.split(","):
            continue
        info = plistlib.loads(
            subprocess.check_output(
                ["diskutil", "info", "-plist", partition.mountpoint],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        )
        devices = {info.get("DeviceIdentifier", "")}
        devices.update(s.get("APFSPhysicalStore", "") for s in info.get("APFSPhysicalStores", []))
        if not devices & physical_devices:
            continue
        group = info.get("APFSVolumeGroupID") or info.get("APFSContainerReference") or partition.mountpoint
        if group in volume_groups:
            continue
        volume_groups.add(group)
        mounts.append(partition.mountpoint)
    return mounts
```
</details>

<br>

### Method `ultralytics.utils.logger._DriveInfo._sort` {#ultralytics.utils.logger.\_DriveInfo.\_sort}

```python
def _sort(mounts)
```

Sort mounted paths with root first, excluding boot/firmware partitions like /boot and /boot/efi.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L337-L340">View on GitHub</a>
```python
@staticmethod
def _sort(mounts):
    """Sort mounted paths with root first, excluding boot/firmware partitions like /boot and /boot/efi."""
    mounts = {m for m in mounts if not (m + "/").startswith(("/boot/", "/efi/"))}
    return sorted(mounts, key=lambda mount: (mount != "/", mount))
```
</details>

<br>

### Method `ultralytics.utils.logger._DriveInfo._windows_mounts` {#ultralytics.utils.logger.\_DriveInfo.\_windows\_mounts}

```python
def _windows_mounts(_partitions)
```

Get Windows fixed local drive mounts.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L415-L427">View on GitHub</a>
```python
@staticmethod
def _windows_mounts(_partitions):
    """Get Windows fixed local drive mounts."""
    output = subprocess.check_output(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object -ExpandProperty DeviceID",
        ],
        text=True,
        timeout=5,
    )
    return [f"{drive}\\" for drive in (line.strip() for line in output.splitlines()) if drive]
```
</details>

<br>

### Method `ultralytics.utils.logger._DriveInfo.mounts` {#ultralytics.utils.logger.\_DriveInfo.mounts}

```python
def mounts(psutil, all_drives=False)
```

Get mounted paths to monitor.

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L311-L334">View on GitHub</a>
```python
@staticmethod
def mounts(psutil, all_drives=False):
    """Get mounted paths to monitor."""
    partitions = [p for p in psutil.disk_partitions(all=False) if p.mountpoint]
    if not all_drives:
        return [_DriveInfo._current_mount(partitions)]

    mounts = [
        p.mountpoint for p in partitions if Path(p.mountpoint).is_dir() and "dontbrowse" not in p.opts.split(",")
    ]
    if len(mounts) <= 1:
        return _DriveInfo._sort(mounts) or [_DriveInfo._current_mount(partitions)]

    for getter in (
        _DriveInfo._macos_mounts if MACOS else None,
        _DriveInfo._linux_mounts if LINUX else None,
        _DriveInfo._windows_mounts if WINDOWS else None,
    ):
        if getter:
            try:
                if platform_mounts := getter(partitions):
                    return _DriveInfo._sort(platform_mounts)
            except (json.JSONDecodeError, OSError, plistlib.InvalidFileException, subprocess.SubprocessError):
                pass
    return _DriveInfo._sort(mounts)
```
</details>


<br><br><hr><br>

## Class `ultralytics.utils.logger.SystemLogger` {#ultralytics.utils.logger.SystemLogger}

```python
SystemLogger(all_drives=False)
```

Log dynamic system metrics for training monitoring.

Captures real-time system metrics including CPU, RAM, disk I/O, network I/O, and NVIDIA GPU statistics for training performance monitoring and analysis.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `all_drives` | `bool` | If True, monitor all mounted drives. If False, monitor the current drive. | `False` |

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

Basic usage (single drive):

```python
>>> logger = SystemLogger()
>>> metrics = logger.get_metrics()
>>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
>>> for disk in metrics["disk"]:
...     print(f"{disk['mount']}: {disk['used_gb']}/{disk['total_gb']} GB")
```

Monitor all drives:

```python
>>> logger = SystemLogger(all_drives=True)
>>> metrics = logger.get_metrics()
>>> for disk in metrics["disk"]:
...     print(f"{disk['mount']}: {disk['used_gb']}/{disk['total_gb']} GB")
```

Training loop integration:

```python
>>> system_logger = SystemLogger()
>>> for epoch in range(epochs):
...     # Training code here
...     metrics = system_logger.get_metrics()
...     # Log to database/file
```

<details>
<summary>Source code in <code>ultralytics/utils/logger.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L430-L637">View on GitHub</a>
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
        Basic usage (single drive):
        >>> logger = SystemLogger()
        >>> metrics = logger.get_metrics()
        >>> print(f"CPU: {metrics['cpu']}%, RAM: {metrics['ram']}%")
        >>> for disk in metrics["disk"]:
        ...     print(f"{disk['mount']}: {disk['used_gb']}/{disk['total_gb']} GB")

        Monitor all drives:
        >>> logger = SystemLogger(all_drives=True)
        >>> metrics = logger.get_metrics()
        >>> for disk in metrics["disk"]:
        ...     print(f"{disk['mount']}: {disk['used_gb']}/{disk['total_gb']} GB")

        Training loop integration:
        >>> system_logger = SystemLogger()
        >>> for epoch in range(epochs):
        ...     # Training code here
        ...     metrics = system_logger.get_metrics()
        ...     # Log to database/file
    """

    def __init__(self, all_drives=False):
        """Initialize the system logger.

        Args:
            all_drives (bool): If True, monitor all mounted drives. If False, monitor the current drive.
        """
        import psutil  # scoped as slow import

        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()
        self.mounts = _DriveInfo.mounts(psutil, all_drives)

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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L615-L637">View on GitHub</a>
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L483-L499">View on GitHub</a>
```python
def _init_nvidia(self):
    """Initialize NVIDIA GPU monitoring with pynvml."""
    if MACOS:
        return False

    try:
        import pynvml  # scoped as slow import

        self.pynvml = pynvml
        pynvml.nvmlInit()
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
def get_metrics(self, rates=False)
```

Get current system metrics including CPU, RAM, disk, network, and GPU usage.

Collects comprehensive system metrics including CPU usage, RAM usage, disk usage, disk I/O statistics, network
I/O statistics, and GPU metrics (if available).

Example output (rates=False, default):
```python
{
    "cpu": 45.2,
    "ram": 78.9,
    "disk": [{"mount": "/", "used_gb": 256.8, "total_gb": 512.0}],
    "disk_io": {"read_mb": 156.7, "write_mb": 89.3},
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
    "disk": [{"mount": "/", "used_gb": 256.8, "total_gb": 512.0}],
    "disk_io": {"read_mbs": 12.5, "write_mbs": 8.3},
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

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/logger.py#L501-L613">View on GitHub</a>
```python
def get_metrics(self, rates=False):
    """Get current system metrics including CPU, RAM, disk, network, and GPU usage.

    Collects comprehensive system metrics including CPU usage, RAM usage, disk usage, disk I/O statistics, network
    I/O statistics, and GPU metrics (if available).

    Example output (rates=False, default):
    ```python
    {
        "cpu": 45.2,
        "ram": 78.9,
        "disk": [{"mount": "/", "used_gb": 256.8, "total_gb": 512.0}],
        "disk_io": {"read_mb": 156.7, "write_mb": 89.3},
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
        "disk": [{"mount": "/", "used_gb": 256.8, "total_gb": 512.0}],
        "disk_io": {"read_mbs": 12.5, "write_mbs": 8.3},
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
    disk_io = psutil.disk_io_counters()
    memory = psutil.virtual_memory()
    now = time.time()

    # Calculate elapsed time since last call
    elapsed = max(0.1, now - self._prev_time)  # Avoid division by zero

    if rates:
        disk_io_metrics = {
            "read_mbs": round(max(0, (disk_io.read_bytes - self._prev_disk.read_bytes) / 1e6 / elapsed), 3),
            "write_mbs": round(max(0, (disk_io.write_bytes - self._prev_disk.write_bytes) / 1e6 / elapsed), 3),
        }
    else:
        disk_io_metrics = {
            "read_mb": round((disk_io.read_bytes - self.disk_start.read_bytes) / 1e6, 3),
            "write_mb": round((disk_io.write_bytes - self.disk_start.write_bytes) / 1e6, 3),
        }

    disks = []
    for mounts in (self.mounts, ["C:\\" if WINDOWS else "/"]):
        for mount in mounts:
            try:
                usage = shutil.disk_usage(mount)
                disks.append(
                    {
                        "mount": mount,
                        "used_gb": round(usage.used / 1e9, 3),
                        "total_gb": round(usage.total / 1e9, 3),
                    }
                )
            except (PermissionError, OSError):
                continue  # Skip inaccessible drives
        if disks:
            break

    metrics = {
        "cpu": round(psutil.cpu_percent(), 3),
        "ram": round(memory.percent, 3),
        "disk": disks,
        "disk_io": disk_io_metrics,
        "gpus": {},
    }

    if rates:
        metrics["network"] = {
            "recv_mbs": round(max(0, (net.bytes_recv - self._prev_net.bytes_recv) / 1e6 / elapsed), 3),
            "sent_mbs": round(max(0, (net.bytes_sent - self._prev_net.bytes_sent) / 1e6 / elapsed), 3),
        }
    else:
        metrics["network"] = {
            "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / 1e6, 3),
            "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / 1e6, 3),
        }

    # Always update previous values for accurate rate calculation on next call
    self._prev_net = net
    self._prev_disk = disk_io
    self._prev_time = now

    # Add GPU metrics (NVIDIA only)
    if self.nvidia_initialized:
        metrics["gpus"].update(self._get_nvidia_metrics())

    return metrics
```
</details>

<br><br>
