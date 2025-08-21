# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import logging
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from time import time

import requests

from ultralytics.utils import RANK

# Initialize default log file
DEFAULT_LOG_PATH = Path("train.log")
if RANK in {-1, 0} and DEFAULT_LOG_PATH.exists():
    DEFAULT_LOG_PATH.unlink(missing_ok=True)


class ConsoleLogger:
    """Console output capture with API/file streaming and deduplication."""

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

        # State tracking
        self.last_line = ""
        self.last_time = 0.0
        self.last_progress_line = ""  # Track 100% progress lines separately
        self.last_was_progress = False  # Track if last line was a progress bar

    def start_capture(self):
        """Start capturing console output."""
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
        """Stop capturing console output."""
        if not self.active:
            return

        self.active = False
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_queue.put(None)

    def _queue_log(self, text):
        """Queue console text with deduplication."""
        if not self.active:
            return

        current_time = time()

        # Handle carriage returns and process lines
        if "\r" in text:
            text = text.split("\r")[-1]

        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()

        for line in lines:
            line = line.rstrip()

            # Handle progress bars - only show 100% completions
            if ("it/s" in line and ("%|" in line or "━" in line)) or (
                "100%" in line and ("it/s" in line or "[" in line)
            ):
                if "100%" not in line:
                    continue
                # Dedupe 100% lines by core content (strip timing)
                progress_core = line.split("[")[0].split("]")[0].strip()
                if progress_core == self.last_progress_line:
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
                payload = {"timestamp": datetime.now().isoformat(), "message": text.strip()}
                requests.post(self.destination, json=payload, timeout=5)
            else:
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                with self.destination.open("a", encoding="utf-8") as f:
                    f.write(text)
        except Exception as e:
            print(f"Platform logging error: {e}", file=self.original_stderr)

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
