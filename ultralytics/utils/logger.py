# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging
import queue
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from ultralytics.utils import MACOS, RANK
from ultralytics.utils.checks import check_requirements

# Initialize default log file
DEFAULT_LOG_PATH = Path("train.log")
if RANK in {-1, 0} and DEFAULT_LOG_PATH.exists():
    DEFAULT_LOG_PATH.unlink(missing_ok=True)


class ConsoleLogger:
    """Console output capture with API/file streaming and deduplication.

    Captures stdout/stderr output and streams it to either an API endpoint or local file, with intelligent deduplication
    to reduce noise from repetitive console output.

    Attributes:
        destination (str | Path): Target destination for streaming (URL or Path object).
        is_api (bool): Whether destination is an API endpoint (True) or local file (False).
        original_stdout: Reference to original sys.stdout for restoration.
        original_stderr: Reference to original sys.stderr for restoration.
        log_queue (queue.Queue): Thread-safe queue for buffering log messages.
        active (bool): Whether console capture is currently active.
        worker_thread (threading.Thread): Background thread for processing log queue.
        last_line (str): Last processed line for deduplication.
        last_time (float): Timestamp of last processed line.
        last_progress_line (str): Last progress bar line for progress deduplication.
        last_was_progress (bool): Whether the last line was a progress bar.

    Examples:
        Basic file logging:
        >>> logger = ConsoleLogger("training.log")
        >>> logger.start_capture()
        >>> print("This will be logged")
        >>> logger.stop_capture()

        API streaming:
        >>> logger = ConsoleLogger("https://api.example.com/logs")
        >>> logger.start_capture()
        >>> # All output streams to API
        >>> logger.stop_capture()
    """

    def __init__(self, destination):
        """Initialize with API endpoint or local file path.

        Args:
            destination (str | Path): API endpoint URL (http/https) or local file path for streaming output.
        """
        self.destination = destination
        self.is_api = isinstance(destination, str) and destination.startswith(("http://", "https://"))
        if not self.is_api:
            self.destination = Path(destination)

        # Console capture
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_queue = queue.Queue(maxsize=1000)
        self.active = False
        self.worker_thread = None

        # State tracking
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_line = ""  # Track last progress line for deduplication
        self.last_was_progress = False  # Track if last line was a progress bar

    def start_capture(self):
        """Start capturing console output and redirect stdout/stderr to custom capture objects."""
        if self.active:
            return

        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)

        # Hook Ultralytics logger
        try:
            handler = self._LogHandler(self._queue_log)
            logging.getLogger("ultralytics").addHandler(handler)
        except Exception:
            pass

        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def stop_capture(self):
        """Stop capturing console output and restore original stdout/stderr."""
        if not self.active:
            return

        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_queue.put(None)

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

            # Deduplicate completed progress bars only if they match the previous progress line
            if " ━━" in line:
                progress_core = line.split(" ━━")[0].strip()
                if progress_core == self.last_progress_line and self.last_was_progress:
                    continue
                self.last_progress_line = progress_core
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

            # Queue with overflow protection
            if not self._safe_put(f"{line}\n"):
                continue  # Skip if queue handling fails

    def _safe_put(self, item):
        """Safely put item in queue with overflow handling."""
        try:
            self.log_queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                self.log_queue.get_nowait()  # Drop oldest
                self.log_queue.put_nowait(item)
                return True
            except queue.Empty:
                return False

    def _stream_worker(self):
        """Background worker for streaming logs to destination."""
        while self.active:
            try:
                log_text = self.log_queue.get(timeout=1)
                if log_text is None:
                    break
                self._write_log(log_text)
            except queue.Empty:
                continue

    def _write_log(self, text):
        """Write log to API endpoint or local file destination."""
        try:
            if self.is_api:
                import requests  # scoped as slow import

                payload = {"timestamp": datetime.now().isoformat(), "message": text.strip()}
                requests.post(str(self.destination), json=payload, timeout=5)
            else:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                with self.destination.open("a", encoding="utf-8") as f:
                    f.write(text)
        except Exception as e:
            print(f"Platform logging error: {e}", file=self.original_stderr)

    class _ConsoleCapture:
        """Lightweight stdout/stderr capture."""

        __slots__ = ("callback", "original")

        def __init__(self, original, callback):
            """Initialize a stream wrapper that redirects writes to a callback while preserving the original."""
            self.original = original
            self.callback = callback

        def write(self, text):
            """Forward text to the wrapped original stream, preserving default stdout/stderr semantics."""
            self.original.write(text)
            self.callback(text)

        def flush(self):
            """Flush the wrapped stream to propagate buffered output promptly during console capture."""
            self.original.flush()

    class _LogHandler(logging.Handler):
        """Lightweight logging handler."""

        __slots__ = ("callback",)

        def __init__(self, callback):
            """Initialize a lightweight logging.Handler that forwards log records to the provided callback."""
            super().__init__()
            self.callback = callback

        def emit(self, record):
            """Format and forward LogRecord messages to the capture callback for unified log streaming."""
            self.callback(self.format(record) + "\n")


class SystemLogger:
    """Log dynamic system metrics for training monitoring.

    Captures real-time system metrics including CPU, RAM, disk I/O, network I/O, and NVIDIA GPU statistics for training
    performance monitoring and analysis.

    Attributes:
        all_drives (bool): Whether to monitor all mounted drives or just the root drive.
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
            all_drives (bool): If True, monitor all mounted drives. If False, only monitor root drive "/".
        """
        import psutil  # scoped as slow import

        self.all_drives = all_drives
        self.pynvml = None
        self.nvidia_initialized = self._init_nvidia()
        self.net_start = psutil.net_io_counters()
        self.disk_start = psutil.disk_io_counters()

        # For rate calculation
        self._prev_net = self.net_start
        self._prev_disk = self.disk_start
        self._prev_time = time.time()

    def _init_nvidia(self):
        """Initialize NVIDIA GPU monitoring with pynvml."""
        try:
            assert not MACOS
            check_requirements("nvidia-ml-py>=12.0.0")
            self.pynvml = __import__("pynvml")
            self.pynvml.nvmlInit()
            return True
        except Exception:
            return False

    def get_metrics(self, rates=False):
        """Get current system metrics including CPU, RAM, disk, network, and GPU usage.

        Collects comprehensive system metrics including CPU usage, RAM usage, disk I/O statistics, network I/O
        statistics, and GPU metrics (if available).

        Example output (rates=False, default):
        ```python
        {
            "cpu": 45.2,
            "ram": 78.9,
            "disk": [{"mount": "/", "read_mb": 156.7, "write_mb": 89.3, "used_gb": 256.8, "total_gb": 512.0}],
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
            "disk": [{"mount": "/", "read_mbs": 12.5, "write_mbs": 8.3, "used_gb": 256.8, "total_gb": 512.0}],
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

        # Calculate disk I/O values (aggregate across all drives)
        if rates:
            read_val = round(max(0, (disk_io.read_bytes - self._prev_disk.read_bytes) / (1 << 20) / elapsed), 3)
            write_val = round(max(0, (disk_io.write_bytes - self._prev_disk.write_bytes) / (1 << 20) / elapsed), 3)
            read_key, write_key = "read_mbs", "write_mbs"
        else:
            read_val = round((disk_io.read_bytes - self.disk_start.read_bytes) / (1 << 20), 3)
            write_val = round((disk_io.write_bytes - self.disk_start.write_bytes) / (1 << 20), 3)
            read_key, write_key = "read_mb", "write_mb"

        # Collect drive usage data as a list (physical volumes only)
        if self.all_drives:
            mounts = []
            for p in psutil.disk_partitions(all=False):
                m = p.mountpoint
                # Include root, /Volumes/* (macOS external), and standard Linux/Windows mounts
                if m == "/" or m.startswith("/Volumes/") or (len(m) == 3 and m[1:] == ":\\"):
                    mounts.append(m)
                # Linux: include /home, /mnt/*, /media/* but not system paths
                elif m in {"/home"} or m.startswith(("/mnt/", "/media/")):
                    mounts.append(m)
        else:
            mounts = ["/"]
        disk = []
        for mount in mounts:
            try:
                usage = shutil.disk_usage(mount)
                disk.append({
                    "mount": mount,
                    read_key: read_val,
                    write_key: write_val,
                    "used_gb": round(usage.used / (1 << 30), 3),
                    "total_gb": round(usage.total / (1 << 30), 3),
                })
            except (PermissionError, OSError):
                continue  # Skip inaccessible drives

        metrics = {
            "cpu": round(psutil.cpu_percent(), 3),
            "ram": round(memory.percent, 3),
            "disk": disk,
            "gpus": {},
        }

        if rates:
            metrics["network"] = {
                "recv_mbs": round(max(0, (net.bytes_recv - self._prev_net.bytes_recv) / (1 << 20) / elapsed), 3),
                "sent_mbs": round(max(0, (net.bytes_sent - self._prev_net.bytes_sent) / (1 << 20) / elapsed), 3),
            }
        else:
            metrics["network"] = {
                "recv_mb": round((net.bytes_recv - self.net_start.bytes_recv) / (1 << 20), 3),
                "sent_mb": round((net.bytes_sent - self.net_start.bytes_sent) / (1 << 20), 3),
            }

        # Always update previous values for accurate rate calculation on next call
        self._prev_net = net
        self._prev_disk = disk_io
        self._prev_time = now

        # Add GPU metrics (NVIDIA only)
        if self.nvidia_initialized:
            metrics["gpus"].update(self._get_nvidia_metrics())

        return metrics

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


if __name__ == "__main__":
    print("SystemLogger Real-time Metrics Monitor")
    print("Press Ctrl+C to stop\n")

    logger = SystemLogger(all_drives=True)

    try:
        while True:
            metrics = logger.get_metrics()

            # Clear screen (works on most terminals)
            print("\033[H\033[J", end="", flush=True)

            # Display system metrics
            print(f"CPU: {metrics['cpu']:5.1f}%")
            print(f"RAM: {metrics['ram']:5.1f}%")
            print(f"Net Recv: {metrics['network']['recv_mb']:9.1f} MB")
            print(f"Net Sent: {metrics['network']['sent_mb']:9.1f} MB")

            # Display disk metrics
            print("\nDisk Metrics:")
            for disk in metrics["disk"]:
                print(
                    f"  {disk['mount']}: {disk['used_gb']:.1f}/{disk['total_gb']:.1f} GB | "
                    f"R: {disk['read_mb']:.1f} MB | W: {disk['write_mb']:.1f} MB"
                )

            # Display GPU metrics if available
            if metrics["gpus"]:
                print("\nGPU Metrics:")
                for gpu_id, gpu_data in metrics["gpus"].items():
                    print(
                        f"  GPU {gpu_id}: {gpu_data['usage']:3}% | "
                        f"Mem: {gpu_data['memory']:5.1f}% | "
                        f"Temp: {gpu_data['temp']:2}°C | "
                        f"Power: {gpu_data['power']:3}W"
                    )
            else:
                print("\nGPU: No NVIDIA GPUs detected")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")
