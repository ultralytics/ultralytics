# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging
import queue
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import time

import requests

from ultralytics.utils import RANK

# Initialize default log file
DEFAULT_LOG_PATH = Path("train.log")
if RANK in {-1, 0}:
    try:
        if DEFAULT_LOG_PATH.exists():
            DEFAULT_LOG_PATH.unlink()
    except Exception:
        pass


class ConsoleLogger:
    """High-performance console output capture with API/file streaming."""

    def __init__(self, destination):
        """Initialize with API endpoint or local file path."""
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
        
        # Deduplication state
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_core = ""
        
        # Compiled regex for performance
        self._progress_pattern = re.compile(r"100%\|[#\s]*\|")
        self._timing_pattern = re.compile(r"\[\d+:\d+<[\d:,]+\s*[\d.]+it/s\]")
        self._timestamp_pattern = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")

    def start_capture(self):
        """Start capturing console output."""
        if self.active:
            return
        
        self.active = True
        sys.stdout = self._ConsoleCapture(self.original_stdout, self._queue_log)
        sys.stderr = self._ConsoleCapture(self.original_stderr, self._queue_log)
        
        self._hook_ultralytics_logger()
        self.worker_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.worker_thread.start()

    def _hook_ultralytics_logger(self):
        """Hook Ultralytics logger to capture log messages."""
        try:
            logger = logging.getLogger("ultralytics")
            handler = self._LogHandler(self._queue_log)
            logger.addHandler(handler)
        except Exception:
            pass

    def stop_capture(self):
        """Stop capturing console output."""
        if not self.active:
            return
        
        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_queue.put(None)  # Stop signal

    def _queue_log(self, text):
        """Queue console text with deduplication."""
        if not self.active:
            return
        
        current_time = time()
        
        # Handle carriage returns
        if "\r" in text:
            text = text.split("\r")[-1]
        
        # Process each line, but skip the last empty element if text ends with newline
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        
        for line in lines:
            line = line.rstrip()
            
            # Check for progress bars (skip empty lines for this check)
            if line:
                is_progress = "it/s" in line and "%|" in line
                if is_progress:
                    if not self._progress_pattern.search(line):
                        continue
                    # Dedupe by core content (strip timing)
                    progress_core = self._timing_pattern.sub("", line).strip()
                    if progress_core == self.last_progress_core:
                        continue
                    self.last_progress_core = progress_core
                else:
                    # Regular deduplication
                    if line == self.last_line and current_time - self.last_time < 0.1:
                        continue
            
            self.last_line = line
            self.last_time = current_time
            
            # Add timestamp (for all lines including empty ones)
            if not self._timestamp_pattern.match(line):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{timestamp}] {line}"
            
            # Queue with overflow protection
            try:
                self.log_queue.put_nowait(f"{line}\n")
            except queue.Full:
                try:
                    self.log_queue.get_nowait()
                    self.log_queue.put_nowait(f"{line}\n")
                except queue.Empty:
                    pass

    def _stream_worker(self):
        """Background worker for streaming logs."""
        while self.active:
            try:
                log_text = self.log_queue.get(timeout=1)
                if log_text is None:
                    break
                self._write_log(log_text)
            except queue.Empty:
                continue

    def _write_log(self, text):
        """Write log to destination."""
        try:
            if self.is_api:
                self._write_api(text)
            else:
                self._write_local(text)
        except Exception as e:
            print(f"Platform logging error: {e}", file=self.original_stderr)

    def _write_api(self, text):
        """Send log to API endpoint."""
        payload = {"timestamp": datetime.now().isoformat(), "message": text.strip()}
        requests.post(self.destination, json=payload, timeout=5)

    def _write_local(self, text):
        """Write log to local file."""
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        with self.destination.open("a", encoding="utf-8") as f:
            f.write(text)

    class _ConsoleCapture:
        """Lightweight stdout/stderr capture."""
        
        __slots__ = ("original", "callback")
        
        def __init__(self, original, callback):
            self.original = original
            self.callback = callback
        
        def write(self, text):
            self.original.write(text)
            self.callback(text)
        
        def flush(self):
            self.original.flush()
    
    class _LogHandler(logging.Handler):
        """Lightweight logging handler."""
        
        __slots__ = ("callback",)
        
        def __init__(self, callback):
            super().__init__()
            self.callback = callback
        
        def emit(self, record):
            self.callback(self.format(record) + "\n")
